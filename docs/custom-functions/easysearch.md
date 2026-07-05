# EasySearch para busqueda web

EasySearch es una Function de tipo `filter` tomada de la comunidad Open WebUI. Su comportamiento original no es el de una tool que el modelo decide invocar: el filtro solo se ejecuta cuando el mensaje del usuario coincide con su disparador.

## Disparadores

- `?? consulta`: busqueda explicita.
- `??`: busqueda basada en el contexto reciente del chat.
- Parche local: solicitudes naturales y conservadoras como `busca por internet`, `ver por internet`, `consulta noticias recientes`, `ultimo resultado`, `marcador`, `precio actual` o `clima`.

## Configuracion local

- La Function queda marcada como `Global` para estar disponible en chats individuales, comparacion y chats grupales.
- El prefijo `??` se mantiene para forzar busqueda cuando el detector automatico no aplique.
- Si se desactiva `Global`, tambien puede activarse por chat desde los controles laterales.

## Requisitos

EasySearch usa el flujo nativo de busqueda web de Open WebUI. Si el backend de Web Search no esta configurado en `Admin Settings > Web Search`, el filtro puede activarse pero no devolver resultados utiles.

## Ejemplos

```text
?? ultimo partido de la seleccion Colombia
puedes buscar por internet como acabo el ultimo partido de la seleccion Colombia?
consulta noticias recientes sobre Open WebUI
```

## Compatibilidad state.config

Parche local `0.4.3-local.2`: algunas versiones recientes de Open WebUI no exponen `request.app.state.config`. EasySearch ahora tolera esa ausencia y delega la configuracion real de busqueda al flujo nativo `process_web_search` / `get_retrieval_config()`.

## Configuracion nativa Web Search

EasySearch solo prepara el mensaje y delega la busqueda al backend nativo de Open WebUI. En desarrollo local quedo configurado:

```text
web.search.enable = true
web.search.engine = duckduckgo
web.search.ddgs_backend = auto
```

Si aparece `Search Error` aun cuando EasySearch se dispara, revisar primero `Admin Settings > Web Search`: el motor debe estar habilitado y seleccionado. DuckDuckGo/DDGS sirve para pruebas porque no requiere API key.
