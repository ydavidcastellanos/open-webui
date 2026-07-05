# Multi Model Conversations v2 Local

Esta carpeta versiona la Function `multi_model_conversations_v2` que usamos como Pipe de chat grupal multimodelo en Open WebUI.

## Archivos

- `custom/functions/multi_model_conversations_v2.py`: codigo Python exportado desde SQLite. Es la fuente de verdad versionable.
- `custom/functions/multi_model_conversations_v2.meta.json`: metadata reproducible de Open WebUI, sin `user_id` local.
- `scripts/import_openwebui_function.py`: upsert de la Function versionada dentro de `backend/data/webui.db` u otra base SQLite compatible.

## Importar En Local

Desde la raiz del repo:

```powershell
C:\Users\caste\Proyecto_Chat\.venv\Scripts\python.exe scripts\import_openwebui_function.py --db backend\data\webui.db
```

Para simular sin escribir:

```powershell
C:\Users\caste\Proyecto_Chat\.venv\Scripts\python.exe scripts\import_openwebui_function.py --dry-run
```

Si la Function no existe aun en una base nueva, puedes asignar propietario:

```powershell
C:\Users\caste\Proyecto_Chat\.venv\Scripts\python.exe scripts\import_openwebui_function.py --user-id <OPEN_WEBUI_USER_ID>
```

## Parches Locales Incluidos

- Configuracion de participantes persistida por chat en `chat.meta.multi_model_config`.
- Popup sin modelos hardcodeados y con lista de modelos disponible desde Open WebUI.
- Ruteo correcto de chips nativos `@modelo`: el transporte sigue siendo `Conversation Pipe` y `selected_model` decide el participante.
- Participantes agregados tarde quedan persistidos y participan en rondas posteriores.
- Limpieza de historial para que los modelos no reciban `details`, `summary` ni warnings `Tools Unavailable` como contenido conversacional.
- Parser de fases para debates: rondas de debate, votacion y conclusion se planifican por solicitud actual, sin heredar votos de problemas anteriores.

## Notas Para Despliegue

- Haz backup de `webui.db` antes de importar en produccion.
- Las Functions ejecutan codigo Python en el servidor; tratar este archivo como codigo privilegiado.
- En produccion, mantener estable `WEBUI_SECRET_KEY` y probar primero en una copia de la base.
