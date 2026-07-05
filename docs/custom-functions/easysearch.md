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
