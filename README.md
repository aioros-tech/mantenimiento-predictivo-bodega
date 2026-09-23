# Mantenimiento Predictivo — Bodega de Atún

Sistema de pronóstico y detección de anomalías para cuartos fríos y blast freezers,
con interpretación automática de resultados en lenguaje natural.

**Dashboard en vivo:** https://aioros-tech.github.io/mantenimiento-predictivo-bodega/

## Cómo funciona

```
Sensores → TimesFM 2.5 → Hechos y comparaciones (código) → SLM → Reporte
```

**TimesFM 2.5** (Google) pronostica la evolución de cada sensor y produce los intervalos
de confianza. El **código** compara contra el umbral operativo y decide el estado. El
**modelo de lenguaje** únicamente redacta el resultado: no calcula, no compara y no
convierte unidades.

Este reparto es deliberado. En una prueba inicial, el modelo de lenguaje afirmó que
89.12 era *superior* a 91.0 y convirtió 32 minutos en «32 horas». Al mover todo el
razonamiento numérico al código, un error del modelo solo puede afectar la redacción,
nunca los datos. Además:

- **Guardarraíl numérico:** un verificador comprueba que cada cifra del texto exista
  en los datos de origen.
- **Veredicto determinista:** el estado (NORMAL / ALERTA) y la tabla los genera el
  código. La información crítica nunca depende del modelo de lenguaje.

## Selección del modelo de pronóstico

Se compararon **Chronos-Bolt** (Amazon) y **TimesFM 2.5** (Google) en modo zero-shot,
contra un baseline naive de persistencia, sobre 60 ventanas móviles.

| Modelo | MAE | MAPE | Cobertura del intervalo 80% |
|---|---|---|---|
| Naive (persistencia) | 0.328 | 0.37% | 80.3% |
| Chronos-Bolt | 0.240 | 0.27% | 73.8% |
| **TimesFM 2.5** | **0.179** | **0.20%** | **79.7%** |

TimesFM 2.5 es el más preciso y el único bien calibrado: su intervalo del 80% cubre
el 79.7% real. Chronos queda sub-cubierto (73.8%), lo que en detección de anomalías
se traduce en falsas alarmas.

## Limitaciones conocidas

- Los datos son del benchmark público **SKAB**, usado como sustituto mientras se
  integra la telemetría real. Los valores están en unidades del sensor, no en °C.
- SKAB dura 2.61 horas: para horizontes de 16 min o más solo queda **una ventana
  independiente**, así que el horizonte operacional (1–6 h) no está validado.
- Todo se midió **sin cuantizar**. El despliegue previsto es sobre Nvidia Jetson
  Orin Nano 8 GB con modelos cuantizados; esas cifras deben revalidarse sobre la placa.
- Los modelos Gemma son de acceso restringido; el sistema está preparado para Gemma
  y se ejecuta mientras tanto con un modelo abierto equivalente.

## Estructura

    index.html                    dashboard publicado
    src/reporte_slm.py            pipeline: pronóstico → hechos → redacción → PDF
    src/comparar_chronos_timesfm.py   comparación de modelos
    src/evaluacion_mejorada.py    métricas de cuantiles y barrido de horizontes
    src/generar_datos_dashboard.py    genera los datos del dashboard
    data/                         dataset SKAB y salida del pipeline
