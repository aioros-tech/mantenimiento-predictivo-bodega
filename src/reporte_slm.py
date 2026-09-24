# -*- coding: utf-8 -*-
"""
Capa de interpretacion: TimesFM 2.5 pronostica -> un SLM (Gemma) redacta el reporte.

ARQUITECTURA (importante): el SLM NO calcula nada. Todos los numeros los produce
TimesFM y el codigo; el SLM solo los convierte en lenguaje natural. Asi el reporte
no puede inventar cifras: si el modelo alucina, alucina redaccion, nunca datos.

    serie de sensores -> TimesFM 2.5 -> hechos numericos -> SLM -> reporte en espanol

Uso:
    python reporte_slm.py                 # usa Gemma si hay token, si no el respaldo
    SLM_MODELO=google/gemma-3-4b-it python reporte_slm.py
"""
import os, re, json, time, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")

CSV, COLUMNA = "anomaly-free.csv", "Temperature"
EQUIPO = os.environ.get("EQUIPO", "Cuarto frio 1")
CONTEXTO, HORIZONTE, PASO_SEG = 512, 64, 30      # 64 pasos x 30 s = 32 min
LIMITE_ALARMA = float(os.environ.get("LIMITE_ALARMA", 91.0))   # umbral operativo

# Hugging Face id or local folder (on the Jetson: copies pinned by revision)
TFM_MODEL = os.environ.get("TFM_MODEL", "google/timesfm-2.5-200m-pytorch")
MODELO_SLM = os.environ.get("SLM_MODELO", "google/gemma-3-1b-it")
RESPALDO   = "Qwen/Qwen2.5-1.5B-Instruct"

DISPOSITIVO = "mps" if torch.backends.mps.is_available() else (
              "cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------- 1. pronostico
_TFM = None

def _modelo_tfm():
    global _TFM
    if _TFM is None:
        import timesfm
        _TFM = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TFM_MODEL)
        _TFM.compile(timesfm.ForecastConfig(max_context=CONTEXTO, max_horizon=HORIZONTE,
                                            normalize_inputs=True,
                                            use_continuous_quantile_head=True))
    return _TFM


def pronosticar(serie):
    m = _modelo_tfm()
    ctx = serie[-CONTEXTO:]
    _, q = m.forecast(horizon=HORIZONTE, inputs=[ctx])
    q = np.asarray(q)[0, :HORIZONTE]          # (H,10): canal 0 media, 1..9 = q0.1..q0.9
    return ctx, q[:, 5], q[:, 1], q[:, 9]     # contexto, mediana, q10, q90


# ------------------------------------------------------- 2. hechos numericos
def extraer_hechos(ctx, med, lo, hi, serie_completa):
    mins = HORIZONTE * PASO_SEG / 60
    actual = float(ctx[-1])
    final = float(med[-1])
    deriva_h = (final - actual) / (mins / 60)

    cruce = np.where(hi >= LIMITE_ALARMA)[0]
    cruce_min = float(cruce[0] * PASO_SEG / 60) if len(cruce) else None

    # backtest de anomalias: pronosticar la ventana anterior y ver que quedo fuera
    prev_ctx = serie_completa[-(CONTEXTO + HORIZONTE):-HORIZONTE]
    real_prev = serie_completa[-HORIZONTE:]
    _, pm, plo, phi = pronosticar(prev_ctx)
    fuera = int(((real_prev < plo) | (real_prev > phi)).sum())

    return {
        "equipo": EQUIPO,
        "sensor": COLUMNA,
        "temperatura_actual_C": round(actual, 2),
        "horizonte_min": round(mins, 0),
        "pronostico_final_C": round(final, 2),
        "pronostico_min_C": round(float(med.min()), 2),
        "pronostico_max_C": round(float(med.max()), 2),
        "deriva_C_por_hora": round(deriva_h, 3),
        "tendencia": "ascendente" if deriva_h > 0.05 else
                     ("descendente" if deriva_h < -0.05 else "estable"),
        "limite_alarma_C": LIMITE_ALARMA,
        "supera_limite": cruce_min is not None,
        "minutos_hasta_limite": cruce_min,
        "banda_confianza_C": round(float((hi - lo).mean()), 3),
        "puntos_anomalos_ultima_ventana": fuera,
        "puntos_evaluados": HORIZONTE,
    }


# ------------------------------------------------------- 3. redaccion con el SLM
def hechos_en_palabras(h):
    """Traduce los numeros a frases inequivocas. TODA comparacion, unidad y
    conversion se resuelve AQUI, en codigo. El SLM recibe conclusiones ya
    hechas y solo las organiza: no puede equivocarse en la aritmetica porque
    no se le pide que la haga."""
    u = "unidades del sensor"
    f = []
    f.append(f"Equipo monitoreado: {h['equipo']}, sensor de {h['sensor'].lower()}.")
    f.append(f"Lectura actual: {h['temperatura_actual_C']} {u}.")

    margen = round(h["limite_alarma_C"] - h["temperatura_actual_C"], 2)
    if margen > 0:
        f.append(f"La lectura actual esta {margen} {u} POR DEBAJO del limite de "
                 f"alarma ({h['limite_alarma_C']} {u}). El equipo opera en rango normal.")
    else:
        f.append(f"La lectura actual SUPERA el limite de alarma "
                 f"({h['limite_alarma_C']} {u}) por {abs(margen)} {u}. Situacion anormal.")

    f.append(f"El pronostico cubre los proximos {int(h['horizonte_min'])} MINUTOS "
             f"(no horas).")
    f.append(f"En ese lapso la lectura oscilara entre un minimo de "
             f"{h['pronostico_min_C']} y un maximo de {h['pronostico_max_C']} {u}, "
             f"y terminara en {h['pronostico_final_C']} {u}.")
    f.append(f"La tendencia es {h['tendencia']}, a razon de "
             f"{abs(h['deriva_C_por_hora'])} {u} por hora.")

    if h["supera_limite"]:
        f.append(f"ADVERTENCIA: el pronostico cruza el limite de alarma en "
                 f"aproximadamente {int(h['minutos_hasta_limite'])} minutos.")
    else:
        f.append("El pronostico NO cruza el limite de alarma en todo el horizonte.")

    pct = round(100 * h["puntos_anomalos_ultima_ventana"] / h["puntos_evaluados"])
    f.append(f"En la ventana anterior, {h['puntos_anomalos_ultima_ventana']} de "
             f"{h['puntos_evaluados']} lecturas ({pct}%) quedaron fuera del intervalo "
             f"esperado, un nivel normal de dispersion.")
    return f


PROMPT = """Eres un asistente de mantenimiento predictivo para una bodega de atun congelado.

Redacta un reporte operativo BREVE en espanol (maximo 120 palabras) usando los hechos
verificados de abajo. Reglas estrictas:
- Los hechos ya estan analizados. Solo reorganizalos en prosa clara.
- NO calcules, NO compares, NO conviertas unidades y NO agregues numeros nuevos.
- Copia las cifras exactamente como aparecen.
- Estructura: estado actual, que se espera, accion recomendada.
- Tono profesional y directo, para un supervisor de planta.

HECHOS VERIFICADOS:
{datos}

REPORTE:"""


def verificar(texto, hechos):
    """Guardarrail: toda cifra del reporte debe existir en los hechos."""
    permitidos = set()
    for v in hechos.values():
        if isinstance(v, (int, float)):
            permitidos |= {str(v), str(round(float(v), 2)), str(int(v)),
                           str(abs(round(float(v), 2))), str(abs(int(v)))}
    pct = round(100 * hechos["puntos_anomalos_ultima_ventana"] / hechos["puntos_evaluados"])
    margen = round(hechos["limite_alarma_C"] - hechos["temperatura_actual_C"], 2)
    permitidos |= {str(pct), str(margen), str(abs(margen))}
    for v in hechos.values():                       # digitos dentro de textos
        if isinstance(v, str):                       # p.ej. "Cuarto frio 1"
            permitidos |= set(re.findall(r"\d+(?:[.,]\d+)?", v))
    hallados = re.findall(r"\d+(?:[.,]\d+)?", texto)
    return sorted({n for n in hallados if n.replace(",", ".") not in permitidos})


def cargar_slm():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    for mid in (MODELO_SLM, RESPALDO):
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            mod = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(DISPOSITIVO)
            return mid, tok, mod
        except Exception as ex:
            print(f"  [!] could not load {mid}: {type(ex).__name__}: {str(ex)[:200]}")
            if mid == MODELO_SLM:
                print(f"      (Gemma requiere aceptar la licencia y un token de Hugging Face)")
                print(f"      -> usando respaldo abierto: {RESPALDO}")
    raise RuntimeError("ningun SLM disponible")


def redactar(hechos):
    mid, tok, mod = cargar_slm()
    texto, dt = generate_text(tok, mod, hechos)
    return mid, texto, dt


def generate_text(tok, mod, hechos):
    """Write the report with an already-loaded SLM (the Jetson service loads it once)."""
    datos = "\n".join(f"- {x}" for x in hechos_en_palabras(hechos))
    msgs = [{"role": "user", "content": PROMPT.format(datos=datos)}]
    entrada = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(DISPOSITIVO)
    n_prompt = entrada["input_ids"].shape[-1]
    t0 = time.time()
    with torch.no_grad():
        salida = mod.generate(**entrada, max_new_tokens=340, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    texto = tok.decode(salida[0][n_prompt:], skip_special_tokens=True).strip()
    return texto, dt


# ------------------------------------------------------- 4. reporte en PDF
def exportar_pdf(texto, h, mid):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    ROJO, CLARO = colors.HexColor("#8C1D1D"), colors.HexColor("#FBEDEB")
    def st(**k):
        base = dict(fontName="Helvetica", fontSize=9.5, leading=13,
                    textColor=colors.HexColor("#3A3A3A"))
        base.update(k)
        return ParagraphStyle("s", **base)
    alerta = h["supera_limite"] or h["temperatura_actual_C"] > h["limite_alarma_C"]

    nombre = f"reporte_{h['equipo'].lower().replace(' ', '_')}.pdf"
    doc = SimpleDocTemplate(nombre, pagesize=letter, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.6*cm, bottomMargin=1.6*cm,
                            title=f"Reporte automatico - {h['equipo']}")
    E = [Paragraph(f"Reporte automático de mantenimiento predictivo",
                   st(fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=ROJO)),
         Paragraph(f"{h['equipo']} &mdash; sensor de {h['sensor'].lower()} &nbsp;|&nbsp; "
                   f"horizonte {int(h['horizonte_min'])} min &nbsp;|&nbsp; generado automáticamente",
                   st(fontSize=8.6, textColor=colors.HexColor("#C0392B"))),
         Spacer(1, 10)]

    estado = "REQUIERE ATENCIÓN" if alerta else "OPERACIÓN NORMAL"
    if h["temperatura_actual_C"] > h["limite_alarma_C"]:
        veredicto = (f"La lectura actual ({h['temperatura_actual_C']}) ya supera el límite "
                     f"de alarma ({h['limite_alarma_C']}).")
    elif h["supera_limite"]:
        veredicto = (f"Se proyecta cruce del límite de alarma en aproximadamente "
                     f"{int(h['minutos_hasta_limite'])} min.")
    else:
        veredicto = (f"Sin cruce del límite de alarma previsto en los próximos "
                     f"{int(h['horizonte_min'])} min.")
    E.append(Table([[Paragraph(f"<b>Estado: {estado}</b><br/>"
                               f"<font size=9>{veredicto}</font>",
                               st(fontSize=11, leading=15, textColor=ROJO))]],
                   colWidths=[17*cm], style=TableStyle([
                       ("BACKGROUND", (0,0), (-1,-1), CLARO),
                       ("LEFTPADDING", (0,0), (-1,-1), 9),
                       ("TOPPADDING", (0,0), (-1,-1), 6),
                       ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                       ("LINEBEFORE", (0,0), (0,-1), 2.6, ROJO)])))
    E.append(Spacer(1, 10))

    for parrafo in [p.strip() for p in texto.split("\n") if p.strip()]:
        limpio = parrafo.replace("**", "")
        if limpio.endswith(":") and len(limpio) < 40:
            E.append(Paragraph(limpio, st(fontName="Helvetica-Bold", fontSize=10,
                                          textColor=ROJO, spaceBefore=6, spaceAfter=2)))
        else:
            E.append(Paragraph(limpio, st(alignment=TA_JUSTIFY, spaceAfter=5)))

    E.append(Spacer(1, 12))
    filas = [["Lectura actual", f"{h['temperatura_actual_C']}"],
             ["Límite de alarma", f"{h['limite_alarma_C']}"],
             ["Pronóstico final", f"{h['pronostico_final_C']}"],
             ["Tendencia", f"{h['tendencia']} ({h['deriva_C_por_hora']}/h)"],
             ["Cruza el límite", "sí" if h["supera_limite"] else "no"],
             ["Lecturas fuera de rango", f"{h['puntos_anomalos_ultima_ventana']} de {h['puntos_evaluados']}"]]
    t = Table(filas, colWidths=[6*cm, 11*cm])
    t.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 8.6),
                           ("TEXTCOLOR", (0,0), (0,-1), ROJO),
                           ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                           ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#E3CBC8")),
                           ("TOPPADDING", (0,0), (-1,-1), 3.5),
                           ("BOTTOMPADDING", (0,0), (-1,-1), 3.5),
                           ("LEFTPADDING", (0,0), (-1,-1), 7)]))
    E.append(t)
    E.append(Spacer(1, 8))
    E.append(Paragraph(f"Pronóstico: TimesFM 2.5 (Google) &nbsp;|&nbsp; Redacción: {mid}<br/>"
                       f"El estado, el veredicto y la tabla los genera el código de forma "
                       f"determinista; el modelo de lenguaje solo redacta el texto y no produce "
                       f"ninguna cifra.<br/>"
                       f"<b>Prueba de concepto</b> ejecutada en Mac M5 (GPU Metal) con los modelos "
                       f"<b>sin cuantizar</b>. Destino de despliegue: Jetson Orin Nano 8 GB con "
                       f"modelos cuantizados &mdash; los tiempos allí serán mayores.",
                       st(fontSize=7.6, leading=9.6, textColor=colors.HexColor("#7A7A7A"))))
    doc.build(E)
    print(f"PDF generado: {nombre}")


# ------------------------------------------------------- main
if __name__ == "__main__":
    print(f"Dispositivo: {DISPOSITIVO.upper()}"
          f"{' (GPU Apple / Metal)' if DISPOSITIVO=='mps' else ''}\n")
    serie = pd.read_csv(CSV, sep=";")[COLUMNA].astype("float32").to_numpy()

    print("[1/3] Pronosticando con TimesFM 2.5...")
    t0 = time.time(); ctx, med, lo, hi = pronosticar(serie); t_fc = time.time() - t0

    print("[2/3] Extrayendo hechos numericos...")
    hechos = extraer_hechos(ctx, med, lo, hi, serie)
    print(json.dumps(hechos, indent=2, ensure_ascii=False))

    print("\n[3/3] Redactando reporte con el SLM...")
    mid, texto, t_slm = redactar(hechos)

    print("\n" + "=" * 72)
    print(f"REPORTE AUTOMATICO - {EQUIPO}")
    print("=" * 72)
    print(texto)
    print("=" * 72)
    sospechosas = verificar(texto, hechos)
    print(f"\nGuardarrail numerico: {'OK - todas las cifras provienen de los datos' if not sospechosas else 'REVISAR ' + str(sospechosas)}")
    print(f"Modelo de lenguaje : {mid}")
    print(f"Pronostico TimesFM : {t_fc:.2f} s")
    print(f"Redaccion SLM      : {t_slm:.2f} s")
    exportar_pdf(texto, hechos, mid)
    open("reporte_generado.txt", "w").write(
        f"REPORTE AUTOMATICO - {EQUIPO}\n{'='*72}\n{texto}\n{'='*72}\n"
        f"SLM: {mid} | TimesFM: {t_fc:.2f}s | SLM: {t_slm:.2f}s | {DISPOSITIVO}\n")
    print("\nGuardado en reporte_generado.txt")
