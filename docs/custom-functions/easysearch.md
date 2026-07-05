# EasySearch para busqueda web

EasySearch es una Function de tipo `filter` tomada de la comunidad Open WebUI. Su comportamiento original no es el de una tool que el modelo decide invocar: el filtro solo se ejecuta cuando el mensaje del usuario coincide con su disparador.

## Disparadores

- `?? consulta`: busqueda explicita.
- `??`: busqueda basada en el contexto reciente del chat.
- Parche local: solicitudes naturales y conservadoras como `busca por internet`, `ver por internet`, `consulta noticias recientes`, `ultimo resultado`, `marcador`, `precio actual` o `clima`.
- Parche local `0.4.3-local.5`: seguimientos deportivos como `cuando juega Colombia?`, `a que hora juega?` o `proximo partido` tambien disparan busqueda aunque no digan `internet` ni `hoy`.

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
cuando juega Colombia?
a que hora juega Brasil?
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

## Uso en comparacion multi-modelo

Parche local `0.4.3-local.3`: Open WebUI puede ejecutar la misma instancia del filtro en paralelo cuando se comparan varios modelos. EasySearch mantiene estado temporal en `self`, por lo que ahora serializa `inlet`/`outlet` para evitar que una busqueda pise el contexto de otra.

El detector automatico tambien reconoce preguntas actuales sin verbo explicito, por ejemplo `quienes son los titulares de Brasil hoy?` o `como va el partido ahora?`.

## Seguimientos y alucinaciones

Parche local `0.4.3-local.5`: en chats de comparacion multi-modelo, una pregunta inicial podia traer fuentes web, pero un seguimiento como `y cuando juega Colombia?` no disparaba EasySearch. Los modelos entonces respondian desde contexto parcial y algunos inferian eliminaciones o calendarios no verificados.

Ahora EasySearch:

- Usa siempre la pregunta original como primera query, antes de cualquier expansion generada por LLM.
- Evita usar `arena-model` para generar queries auxiliares.
- Reconoce preguntas de calendario deportivo con `cuando`, `hora`, `fecha`, `proximo`, `juega`, `partido`, `vs` o `seleccion`.
- Inyecta una instruccion de verificacion: si los resultados no sostienen el dato actual, el modelo debe decir que no pudo verificarlo y no inventar marcadores, alineaciones, horarios ni eliminaciones.

## Deadlock en chats normales y grupales

Parche local `0.4.3-local.6`: EasySearch mantenia un lock mientras llamaba internamente a `generate_chat_completion` para generar queries o extraer contexto. Como EasySearch es global, esa llamada podia reentrar al mismo filtro y quedarse esperando el lock, dejando la respuesta `done = 0` sin contenido final.

Ahora las llamadas internas usan `bypass_filter=True`, restauran el estado original del request al terminar y tienen un guard de reentrada (`easysearch_internal_call`) para devolver el payload sin procesarlo si una llamada interna llegara de nuevo a la cadena de filtros.

## Ajustes recomendados para calidad

Para usar EasySearch como filtro de busqueda y no como RAG nativo de Open WebUI, en desarrollo local se usan estos valores:

```text
web.search.enable = true
web.search.engine = duckduckgo
web.search.ddgs_backend = auto
web.search.result_count = 8
web.search.bypass_web_loader = true
web.search.bypass_embedding_and_retrieval = true
```

Esto deja que Open WebUI solo entregue candidatos de busqueda y que EasySearch haga su propio fetch/limpieza/citas. Tambien evita dependencias innecesarias de loader, embeddings y vector DB en chats de comparacion multi-modelo.
