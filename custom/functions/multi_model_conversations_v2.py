"""
title: Multi Model Conversations v2
author: Haervwe
author_url: https://github.com/Haervwe
funding_url: https://github.com/Haervwe/open-webui-tools
version: 2.9.1
required_open_webui_version: 0.9.1
"""

import logging
import asyncio
import json
import re
import html as html_module
import ast
from uuid import uuid4
from typing import Callable, Awaitable, Any, Optional
import time
import unicodedata
from pydantic import BaseModel, Field
from open_webui.constants import TASKS
from open_webui.main import generate_chat_completions
from open_webui.models.users import User, Users
from open_webui.models.models import Models
from open_webui.internal.db import get_async_db_context
from open_webui.models.chats import Chat
from sqlalchemy.orm.attributes import flag_modified
from open_webui.utils.tools import get_tools, get_builtin_tools, get_terminal_tools
from open_webui.utils.middleware import process_tool_result
from open_webui.utils.chat import (
    generate_chat_completion as generate_raw_chat_completion,
)

name = "Conversation"


def setup_logger():
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name(name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()

THINK_OPEN_PATTERN = re.compile(
    r"<(?:think|thinking|reason|reasoning|thought|Thought)>|\|begin_of_thought\|",
    re.IGNORECASE,
)
THINK_CLOSE_PATTERN = re.compile(
    r"</(?:think|thinking|reason|reasoning|thought|Thought)>|\|end_of_thought\|",
    re.IGNORECASE,
)

SPEAKER_COLORS = ["🔴", "🔵", "🟢", "🟡", "🟣", "🟠", "🟤", "⚫", "⚪"]


def clean_thinking_tags(message: str) -> str:
    complete_pattern = re.compile(
        r"<(think|thinking|reason|reasoning|thought|Thought)>.*?</\1>"
        r"|"
        r"\|begin_of_thought\|.*?\|end_of_thought\|"
        r"|"
        r"<details\s+type=[\"']reasoning[\"'][^>]*>.*?</details>",
        re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(complete_pattern, "", message)

    orphan_close_pattern = re.compile(
        r"</(?:think|thinking|reason|reasoning|thought|Thought)>"
        r"|"
        r"\|end_of_thought\|",
        re.IGNORECASE,
    )

    last_match_end = -1
    for match in orphan_close_pattern.finditer(cleaned):
        last_match_end = match.end()

    if last_match_end != -1:
        cleaned = cleaned[last_match_end:]

    orphan_open_pattern = re.compile(
        r"<(?:think|thinking|reason|reasoning|thought|Thought)>"
        r"|"
        r"\|begin_of_thought\|"
        r"|"
        r"<details[^>]*>",
        re.IGNORECASE,
    )
    cleaned = re.sub(orphan_open_pattern, "", cleaned)

    return cleaned.strip()


class Pipe:
    __current_event_emitter__: Callable[[dict], Awaitable[None]]
    __user__: Optional[User]
    __model__: str

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the pipe operations.",
        )

    class UserValves(BaseModel):
        NUM_PARTICIPANTS: int = Field(
            default=2,
            description="Number of participants in the conversation (1-5)",
            ge=1,
            le=5,
        )
        ROUNDS_PER_CONVERSATION: int = Field(
            default=1, description="Number of rounds in the entire conversation", ge=1
        )
        Participant1Model: str = Field(
            default="", description="Model ID for Participant 1"
        )
        Participant1Alias: str = Field(
            default="", description="Alias for Participant 1"
        )
        Participant1SystemMessage: str = Field(
            default="", description="System Message for Participant 1"
        )
        Participant2Model: str = Field(
            default="", description="Model ID for Participant 2"
        )
        Participant2Alias: str = Field(
            default="", description="Alias for Participant 2"
        )
        Participant2SystemMessage: str = Field(
            default="", description="System Message for Participant 2"
        )
        Participant3Model: str = Field(
            default="", description="Model ID for Participant 3"
        )
        Participant3Alias: str = Field(
            default="", description="Alias for Participant 3"
        )
        Participant3SystemMessage: str = Field(
            default="", description="System Message for Participant 3"
        )
        Participant4Model: str = Field(
            default="", description="Model ID for Participant 4"
        )
        Participant4Alias: str = Field(
            default="", description="Alias for Participant 4"
        )
        Participant4SystemMessage: str = Field(
            default="", description="System Message for Participant 4"
        )
        Participant5Model: str = Field(
            default="", description="Model ID for Participant 5"
        )
        Participant5Alias: str = Field(
            default="", description="Alias for Participant 5"
        )
        Participant5SystemMessage: str = Field(
            default="", description="System Message for Participant 5"
        )
        AllParticipantsApendedMessage: str = Field(
            default="Respond only as your specified character and never use your name as title, just output the response as if you really were talking(no one says his name before a phrase), do not go off character in any situation, Your acted response as",
            description="Appended message to all participants internally to prime them properly",
        )
        UseGroupChatManager: bool = Field(
            default=False,
            description="Use Group Chat Manager to select speakers dynamically",
        )
        ManagerModel: str = Field(
            default="",
            description="Model for the Manager (leave empty to use user's default model)",
        )
        ManagerSystemMessage: str = Field(
            default="You are a group chat manager. Your role is to decide who should speak next in a multi-participant conversation. You will be given the conversation history and a list of participant aliases. Choose the alias of the participant who is most likely to provide a relevant and engaging response to the latest message. Consider the context of the conversation, the personalities of the participants, and avoid repeatedly selecting the same participant.",
            description="System message for the Manager",
        )
        ManagerSelectionPrompt: str = Field(
            default="Conversation History:\n{history}\n\nThe last speaker was '{last_speaker}'. Based on the flow of the conversation, who should speak next? Choose exactly one from the following list of participants: {participant_list}\n\nRespond with ONLY the alias of your choice, and nothing else.",
            description="Template for the Manager's selection prompt. Use {history}, {last_speaker}, and {participant_list}.",
        )
        Temperature: float = Field(default=1, description="Models temperature")
        Top_k: int = Field(default=50, description="Models top_k")
        Top_p: float = Field(default=0.8, description="Models top_p")

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        return [{"id": f"{name}-pipe", "name": f"{name} Pipe"}]

    async def _load_history_with_model_from_chat(self) -> Optional[list[dict]]:
        chat_id = self.__metadata__.get("chat_id")
        message_id = self.__metadata__.get("user_message_id")
        if not chat_id or not message_id or str(chat_id).startswith(("local:", "channel:")):
            return None

        try:
            async with get_async_db_context() as session:
                chat = await session.get(Chat, chat_id)
                if not chat:
                    return None
                history = (chat.chat or {}).get("history", {})
                messages = history.get("messages", {})
                if not isinstance(messages, dict):
                    return None

                chain = []
                current = messages.get(message_id)
                visited = set()
                while isinstance(current, dict) and current.get("id") not in visited:
                    visited.add(current.get("id"))
                    chain.append(current)
                    parent_id = current.get("parentId")
                    current = messages.get(parent_id) if parent_id else None

                if not chain:
                    return None

                chain.reverse()
                allowed = {"role", "content", "output", "files", "contextSummary", "model"}
                return [
                    {key: value for key, value in message.items() if key in allowed}
                    for message in chain
                ]
        except Exception as e:
            logger.error(f"Failed to load model-aware chat history: {e}")
            return None

    def _extract_config_from_metadata(self, body: dict) -> Optional[dict]:
        containers = []

        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            containers.append(metadata)

        chat_metadata = body.get("chat_metadata")
        if isinstance(chat_metadata, dict):
            containers.append(chat_metadata)

        params = body.get("params")
        if isinstance(params, dict):
            params_metadata = params.get("metadata")
            if isinstance(params_metadata, dict):
                containers.append(params_metadata)

        for container in containers:
            config = container.get("multi_model_config")
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except json.JSONDecodeError:
                    continue
            if isinstance(config, dict):
                return config
        return None

    def _persist_config_to_metadata(self, body: dict, config: dict) -> None:
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            body["metadata"] = metadata
        metadata["multi_model_config"] = config

    async def _load_config_from_chat_meta(self) -> Optional[dict]:
        chat_id = self.__metadata__.get("chat_id")
        if not chat_id or str(chat_id).startswith(("local:", "channel:")):
            return None
        try:
            async with get_async_db_context() as session:
                chat = await session.get(Chat, chat_id)
                if not chat:
                    return None
                meta = chat.meta or {}
                config = meta.get("multi_model_config")
                if isinstance(config, str):
                    try:
                        config = json.loads(config)
                    except json.JSONDecodeError:
                        return None
                return config if isinstance(config, dict) else None
        except Exception as e:
            logger.error(f"Failed to load multi-model config from chat meta: {e}")
            return None

    async def _persist_config_to_chat_meta(self, config: dict) -> None:
        chat_id = self.__metadata__.get("chat_id")
        if not chat_id or str(chat_id).startswith(("local:", "channel:")):
            return
        try:
            async with get_async_db_context() as session:
                chat = await session.get(Chat, chat_id)
                if not chat:
                    return
                meta = dict(chat.meta or {})
                meta["multi_model_config"] = config
                chat.meta = meta
                flag_modified(chat, "meta")
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to persist multi-model config to chat meta: {e}")

    def _build_default_config_from_valves(self, valves) -> dict:
        # New chats should be explicitly configured in the popup instead of
        # inheriting hard-coded participants. Existing chats still load their
        # saved config from chat.meta["multi_model_config"].
        return {
            "rounds": 1,
            "use_manager": bool(getattr(valves, "UseGroupChatManager", False)),
            "manager_model": getattr(valves, "ManagerModel", ""),
            "participants": [],
        }

    def _sanitize_config(self, config: Optional[dict], valves) -> dict:
        if not isinstance(config, dict):
            return self._build_default_config_from_valves(valves)

        participants = []
        for participant in config.get("participants", []):
            if not isinstance(participant, dict):
                continue
            model = str(participant.get("model", "")).strip()
            if not model:
                continue
            alias = str(participant.get("alias", "")).strip() or model
            system_message = str(participant.get("system_message", "")).strip()
            participants.append(
                {
                    "model": model,
                    "alias": alias,
                    "system_message": system_message,
                }
            )

        # The popup config defines participants only. Keep the persisted baseline
        # at one round; per-message round counts are parsed from David's prompt.
        rounds = 1

        use_manager = bool(config.get("use_manager", valves.UseGroupChatManager))

        manager_model = str(
            config.get("manager_model", "") or valves.ManagerModel or ""
        ).strip()

        return {
            "rounds": rounds,
            "use_manager": use_manager,
            "manager_model": manager_model,
            "participants": participants,
        }

    def _clean_history_content(self, content) -> str:
        """Remove UI/control markup before passing history back to models."""
        if content is None:
            return ""
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)

        cleaned = html_module.unescape(content)
        cleaned = re.sub(
            r"<details\b[^>]*>.*?</details>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"<summary\b[^>]*>.*?</summary>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(r"</?(?:details|summary)[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"Model\s+[\w.\-:]+\s+does\s+not\s+have\s+native\s+function\s+calling\s+enabled\.\s+Tools\s+skipped\s+for\s+[\w.\-:]+\.",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _adapt_history(self, conversation_history: list, alias: str) -> list:
        """
        Adapts the conversation history for a specific participant (alias).
        - Maps other participants' responses to the 'user' role.
        - Preserves own responses as 'assistant'.
        - Ensures strictly alternating user/assistant turns by merging consecutive roles.
        """
        adapted_history = []
        alias_norm = self._normalize_alias(alias)

        for msg in conversation_history:
            speaker = msg.get("_speaker")
            speaker_norm = self._normalize_alias(speaker) if speaker else None

            role = msg.get("role")
            content = self._clean_history_content(msg.get("content", ""))
            if not content and role != "tool":
                continue

            # Determine effective role and content for this participant
            if speaker_norm:
                if speaker_norm != alias_norm:
                    # Other participant's response -> present as user message
                    role = "user"
                    content = f"{speaker} says: {content}"
                else:
                    # This participant's own past response -> assistant role
                    role = "assistant"
                    # Include private tool interaction chain if any
                    tool_msgs = msg.get("_tool_messages", [])
                    for tm in tool_msgs:
                        # Add tool messages as-is (they provide naturally alternating turns)
                        tm_cleaned = {
                            k: v for k, v in tm.items() if not k.startswith("_")
                        }
                        adapted_history.append(tm_cleaned)

            # STRICT TURN ALTERNATION: Merge consecutive messages of the same role
            # (Excluding 'tool' messages which can be consecutive and expect specific pairing)
            if (
                adapted_history
                and adapted_history[-1]["role"] == role
                and role not in ["tool"]
                and not adapted_history[-1].get("tool_calls")
            ):
                # Merge content with previous message of the same role
                prev_content = adapted_history[-1].get("content") or ""
                if prev_content:
                    adapted_history[-1]["content"] = prev_content + f"\n\n{content}"
                else:
                    adapted_history[-1]["content"] = content
                continue

            adapted_history.append({"role": role, "content": content})

        return adapted_history

    def _normalize_alias(self, alias: str) -> str:
        text = unicodedata.normalize("NFKD", (alias or "").lower())
        text = text.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip()
        return re.sub(r"\s+", " ", cleaned)

    def _latest_user_text(self, conversation_history: list) -> str:
        for msg in reversed(conversation_history):
            if msg.get("role") == "user":
                return str(msg.get("content", "") or "")
        return ""

    def _should_open_config_popup(self, text: str) -> bool:
        normalized = self._normalize_alias(text)
        config_cues = [
            "configurar modelos",
            "configura modelos",
            "cambiar modelos",
            "cambia modelos",
            "elegir modelos",
            "escoger modelos",
            "seleccionar modelos",
            "configurar participantes",
            "cambiar participantes",
            "elegir participantes",
            "escoger participantes",
            "seleccionar participantes",
            "abrir popup",
            "abrir configuracion",
            "reconfigurar chat",
        ]
        return any(cue in normalized for cue in config_cues)

    def _build_popup_defaults(self, valves, config: Optional[dict] = None) -> dict:
        defaults = {}
        for attr, fieldInfo in valves.model_fields.items():
            defaults[attr] = getattr(valves, attr)

        for i in range(1, 6):
            defaults[f"Participant{i}Model"] = ""
            defaults[f"Participant{i}Alias"] = ""
            defaults[f"Participant{i}SystemMessage"] = ""
        defaults["NUM_PARTICIPANTS"] = 2
        defaults["ROUNDS_PER_CONVERSATION"] = 1

        if isinstance(config, dict):
            participants = config.get("participants", [])
            if isinstance(participants, list) and participants:
                defaults["NUM_PARTICIPANTS"] = min(5, max(1, len(participants)))
                for i, participant in enumerate(participants[:5], start=1):
                    if not isinstance(participant, dict):
                        continue
                    defaults[f"Participant{i}Model"] = participant.get("model", "")
                    defaults[f"Participant{i}Alias"] = (
                        participant.get("alias", "") or participant.get("model", "")
                    )
                    defaults[f"Participant{i}SystemMessage"] = participant.get(
                        "system_message", ""
                    )
            defaults["ROUNDS_PER_CONVERSATION"] = 1
            if "use_manager" in config:
                defaults["UseGroupChatManager"] = bool(config.get("use_manager"))
            if config.get("manager_model"):
                defaults["ManagerModel"] = config.get("manager_model")

        return defaults

    def _participant_match_keys(self, participant: dict) -> list[str]:
        values = [participant.get("alias", ""), participant.get("model", "")]
        keys = []
        for value in values:
            normalized = self._normalize_alias(str(value))
            if normalized:
                keys.append(normalized)
                compact = normalized.replace(" ", "")
                if compact:
                    keys.append(compact)
                first_token = normalized.split(" ")[0]
                if len(first_token) >= 3:
                    keys.append(first_token)
        deduped = []
        for key in keys:
            if key and key not in deduped:
                deduped.append(key)
        return deduped

    def _is_group_request(self, text: str) -> bool:
        normalized = self._normalize_alias(text)
        group_cues = [
            "todos",
            "cada uno",
            "equipo",
            "grupo",
            "ronda",
            "rondas",
            "debate",
            "debatir",
            "dialogo",
            "dialoguen",
            "conversen",
            "entre ustedes",
            "saludarse",
            "saludense",
            "lluvia de ideas",
            "brainstorm",
            "opinan",
            "despedida",
            "despedirse",
            "despidanse",
            "agradezcanse",
        ]
        return any(cue in normalized for cue in group_cues)




    def _requested_rounds(self, text: str) -> Optional[int]:
        normalized = self._normalize_alias(text)
        phase_normalized = re.sub(
            r"\bconclusion\s+(?:anterior|previa|pasada)\b",
            "referencia anterior",
            normalized,
        )

        word_numbers = {
            "una": 1,
            "un": 1,
            "unica": 1,
            "unico": 1,
            "dos": 2,
            "tres": 3,
            "cuatro": 4,
            "cinco": 5,
        }

        def parse_number(raw: str) -> Optional[int]:
            if not raw:
                return None
            raw = raw.strip()
            if raw.isdigit():
                return int(raw)
            return word_numbers.get(raw)

        number_pattern = r"(\d{1,2}|una|un|unica|unico|dos|tres|cuatro|cinco)"
        requested = None

        explicit_rounds = [
            parse_number(value)
            for value in re.findall(
                rf"\b{number_pattern}\s+(?:unica|unico|sola|solo)?\s*(?:ronda|rondas|round|rounds|turno|turnos)\b",
                normalized,
            )
        ]
        explicit_rounds = [value for value in explicit_rounds if value is not None]
        if explicit_rounds:
            requested = max(explicit_rounds)

        vote_requested = any(
            cue in phase_normalized
            for cue in ["votar", "voten", "votacion", "voto", "escojan", "elijan"]
        )
        conclusion_requested = any(
            cue in phase_normalized for cue in ["conclusion", "concluir", "sintesis"]
        )

        additional_rounds = [
            parse_number(value)
            for value in re.findall(
                rf"\b{number_pattern}\s+(?:ronda|rondas|round|rounds|turno|turnos)\s+mas\b",
                normalized,
            )
        ]
        additional_rounds = [value for value in additional_rounds if value is not None]

        initial_one_round_requested = bool(
            re.search(r"\b(?:una|un|unica|unico)\s+(?:unica|unico|sola|solo)?\s*(?:ronda|round|turno)\b", normalized)
        )
        if additional_rounds and initial_one_round_requested:
            # Natural wording like "una ronda mas" means one next round, not
            # the current turn plus one extra cycle.
            requested = max(requested or 0, max(additional_rounds))

        ordinal_numbers = {
            "primera": 1,
            "primer": 1,
            "segunda": 2,
            "segundo": 2,
            "tercera": 3,
            "tercer": 3,
            "cuarta": 4,
            "cuarto": 4,
            "quinta": 5,
            "quinto": 5,
        }
        ordinal_requested = None
        for word, value in ordinal_numbers.items():
            if re.search(rf"\b{word}\s+ronda\b", normalized):
                ordinal_requested = max(ordinal_requested or 0, value)
        digit_ordinals = [
            int(value)
            for value in re.findall(r"\b([1-5])(?:a|o)?\s+ronda\b", normalized)
        ]
        if digit_ordinals:
            ordinal_requested = max(ordinal_requested or 0, max(digit_ordinals))
        if ordinal_requested:
            requested = max(requested or 0, ordinal_requested)

        # Composite requests such as:
        # "2 rondas de debate, una ronda para votar y una ronda final"
        # mean 2 debate rounds + 1 vote + 1 conclusion = 4 total phases.
        debate_match = re.search(
            rf"\b{number_pattern}\s+(?:unica|unico|sola|solo)?\s*"
            rf"(?:ronda|rondas|round|rounds|turno|turnos)\b"
            rf"(?=[a-z0-9\s]{{0,140}}\b(?:debate|debatir|lluvia\s+de\s+ideas|brainstorm)\b)",
            normalized,
        )
        if debate_match:
            debate_rounds = parse_number(debate_match.group(1)) or 0
            composite_total = debate_rounds
            if vote_requested:
                composite_total += 1
            if conclusion_requested:
                composite_total += 1
            if composite_total:
                requested = max(requested or 0, composite_total)

        if (
            requested == 1
            and vote_requested
            and conclusion_requested
            and any(cue in normalized for cue in ["debate", "por modelo", "todos", "cada uno", "equipo"])
        ):
            requested = 3

        if requested is None:
            if vote_requested and conclusion_requested:
                requested = 2
            elif vote_requested or conclusion_requested:
                requested = 1

        if requested is None:
            return None

        return max(1, min(5, requested))

    def _round_plan(self, text: str, rounds: int) -> list[str]:
        normalized = self._normalize_alias(text)
        phase_normalized = re.sub(
            r"\bconclusion\s+(?:anterior|previa|pasada)\b",
            "referencia anterior",
            normalized,
        )
        plan = ["debate"] * max(1, rounds)
        vote_requested = any(
            cue in phase_normalized
            for cue in ["votar", "voten", "votacion", "voto", "escojan", "elijan"]
        )
        conclusion_requested = any(
            cue in phase_normalized for cue in ["conclusion", "concluir", "sintesis"]
        )
        if vote_requested and conclusion_requested and len(plan) >= 2:
            plan[-2] = "vote"
            plan[-1] = "conclusion"
        elif conclusion_requested and plan:
            plan[-1] = "conclusion"
        elif vote_requested and plan:
            plan[-1] = "vote"
        return plan

    def _phase_instruction(self, phase: str, round_number: int, rounds: int, alias: str) -> str:
        if phase == "vote":
            return (
                f"Moderator control note: this is round {round_number} of {rounds}, the voting/selection round. "
                f"Speak only as {alias}. Cast only your vote for which participant should write the final conclusion, "
                "with a brief reason. Do not write the final conclusion and do not continue the debate."
            )
        if phase == "conclusion":
            return (
                f"Moderator control note: this is round {round_number} of {rounds}, the final conclusion round. "
                f"You are the selected participant. Speak only as {alias} and write the final synthesis/conclusion. "
                "Do not start a new debate or ask for more votes."
            )
        return (
            f"Moderator control note: this is round {round_number} of {rounds}, a debate round. "
            f"Speak only as {alias} for the current round. Engage the previous arguments and David's request. "
            "Do not vote, do not choose the final speaker, and do not write a final conclusion yet."
        )

    def _available_model_entries(self) -> list[dict]:
        request = getattr(self, "__request__", None)
        app = getattr(request, "app", None)
        state = getattr(app, "state", None)
        models_state = getattr(state, "MODELS", {}) or {}
        entries = []
        for model_id, model_info in models_state.items():
            if not isinstance(model_info, dict):
                continue
            pipeline = model_info.get("pipeline") or model_info.get("pipe") or {}
            if isinstance(pipeline, dict) and pipeline.get("type") == "filter":
                continue
            model_norm = self._normalize_alias(model_id)
            if not model_norm or "conversation pipe" in model_norm:
                continue
            entries.append(
                {
                    "id": model_id,
                    "name": model_info.get("name") or model_id,
                }
            )
        return entries

    def _is_conversation_pipe_model(self, model_id: str) -> bool:
        model_norm = self._normalize_alias(str(model_id or ""))
        return not model_norm or "conversation pipe" in model_norm

    def _model_target_from_label(
        self, label: str, allow_fallback: bool = False
    ) -> Optional[dict]:
        label = str(label or "").strip()
        label_norm = self._normalize_alias(label)
        label_compact = label_norm.replace(" ", "")
        if not label_norm or self._is_conversation_pipe_model(label):
            return None

        for model in self._available_model_entries():
            for value in [model.get("id", ""), model.get("name", "")]:
                value_norm = self._normalize_alias(str(value))
                if not value_norm:
                    continue
                if (
                    label_norm == value_norm
                    or label_compact == value_norm.replace(" ", "")
                ):
                    return {
                        "model": model["id"],
                        "alias": model.get("name") or model["id"],
                        "system_message": "",
                    }

        if allow_fallback:
            return {"model": label, "alias": label, "system_message": ""}
        return None

    def _extract_text_from_output(self, output) -> str:
        if not output:
            return ""

        if isinstance(output, str):
            stripped = output.strip()
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(stripped)
                    if parsed is not output:
                        parsed_text = self._extract_text_from_output(parsed)
                        if parsed_text:
                            return parsed_text
                except Exception:
                    pass
            return stripped

        if isinstance(output, list):
            message_texts = [
                self._extract_text_from_output(item)
                for item in output
                if isinstance(item, dict)
                and (item.get("type") == "message" or item.get("role") == "assistant")
            ]
            message_texts = [text for text in message_texts if text]
            if message_texts:
                return "\n\n".join(message_texts)

            texts = [self._extract_text_from_output(item) for item in output]
            return "\n\n".join(text for text in texts if text)

        if isinstance(output, dict):
            if output.get("type") == "output_text" and isinstance(output.get("text"), str):
                return output["text"]

            if output.get("type") == "message" or output.get("role") == "assistant":
                text = self._extract_text_from_output(output.get("content"))
                if text:
                    return text

            for key in ("message", "delta", "content", "text"):
                value = output.get(key)
                if value:
                    text = self._extract_text_from_output(value)
                    if text:
                        return text

        return ""

    def _participant_known(self, candidate: dict, participants: list[dict]) -> bool:
        candidate_values = [
            self._normalize_alias(candidate.get("model", "")),
            self._normalize_alias(candidate.get("alias", "")),
        ]
        candidate_values = [v for v in candidate_values if v]
        for participant in participants:
            known_values = [
                self._normalize_alias(participant.get("model", "")),
                self._normalize_alias(participant.get("alias", "")),
            ]
            known_values = [v for v in known_values if v]
            if any(value in known_values for value in candidate_values):
                return True
        return False

    def _available_model_targets_from_text(
        self, text: str, participants: list[dict]
    ) -> list[dict]:
        if not text:
            return []
        without_user_prefix = text
        if not re.match(r"^\s*@", without_user_prefix):
            without_user_prefix = re.sub(r"^\s*[^:\n]{1,40}:\s*", "", without_user_prefix, count=1)
        normalized = self._normalize_alias(without_user_prefix)
        compact = normalized.replace(" ", "")
        raw_lower = without_user_prefix.lower()
        known_keys = set()
        for participant in participants:
            for participant_key in self._participant_match_keys(participant):
                known_keys.add(participant_key)
                known_keys.add(participant_key.replace(" ", ""))
        targets = []
        for model in self._available_model_entries():
            values = [model["id"], model.get("name", "")]
            keys = []
            for value in values:
                key = self._normalize_alias(str(value))
                if not key:
                    continue
                keys.append(key)
                keys.append(key.replace(" ", ""))
                first = key.split(" ")[0]
                if len(first) >= 3:
                    keys.append(first)
            for key in dict.fromkeys(keys):
                key_compact = key.replace(" ", "")
                if key in known_keys or key_compact in known_keys:
                    continue
                starts_direct = normalized == key or normalized.startswith(key + " ")
                starts_compact = compact == key_compact or compact.startswith(key_compact)
                tagged = re.search(rf"@{re.escape(key_compact)}\b", raw_lower)
                addressed = re.search(rf"\bpara\s+{re.escape(key)}\b", normalized)
                if starts_direct or starts_compact or tagged or addressed:
                    candidate = {
                        "model": model["id"],
                        "alias": model.get("name") or model["id"],
                        "system_message": "",
                    }
                    if not self._participant_known(candidate, participants + targets):
                        targets.append(candidate)
                    break
        return targets

    def _infer_participants_from_history(
        self, conversation_history: list, participants: list[dict]
    ) -> list[dict]:
        inferred = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                targets = self._available_model_targets_from_text(
                    str(msg.get("content", "") or ""), participants + inferred
                )
            elif msg.get("role") == "assistant":
                targets = []
                for label, allow_fallback in [
                    (msg.get("_speaker"), False),
                    (msg.get("model"), True),
                ]:
                    target = self._model_target_from_label(label, allow_fallback)
                    if target and not self._participant_known(
                        target, participants + inferred + targets
                    ):
                        targets.append(target)
            else:
                continue

            for target in targets:
                if not self._participant_known(target, participants + inferred):
                    inferred.append(target)
        return inferred



    def _select_conclusion_participant(
        self, conversation_history: list, participants: list[dict]
    ) -> dict:
        scores = {index: 0 for index, _ in enumerate(participants)}
        last_vote_order = {}

        vote_intro_pattern = re.compile(
            r"\b(?:mi\s+voto\s+(?:es|es\s+para|va\s+para)|voto\s+por|voto\s+para|elijo|escojo)\b"
        )

        def specific_vote_keys(participant: dict) -> list[str]:
            values = [
                str(participant.get("alias", "") or ""),
                str(participant.get("model", "") or ""),
            ]
            keys = []
            for value in values:
                normalized = self._normalize_alias(value)
                compact = normalized.replace(" ", "")
                # Full aliases/models are specific enough to distinguish similar
                # participants such as kimi-k2.6 and kimi-k2.7-code.
                for key in [normalized, compact]:
                    if key and len(key) >= 4 and key not in keys:
                        keys.append(key)
            return sorted(keys, key=len, reverse=True)

        for msg in conversation_history:
            speaker = msg.get("_speaker")
            if not speaker:
                continue

            text = self._normalize_alias(str(msg.get("content", "") or ""))
            vote_match = vote_intro_pattern.search(text)
            if not vote_match:
                continue

            speaker_norm = self._normalize_alias(speaker)
            vote_text = text[vote_match.end() : vote_match.end() + 220]
            vote_text_compact = vote_text.replace(" ", "")

            best_index = None
            best_len = -1
            for index, participant in enumerate(participants):
                if self._normalize_alias(participant.get("alias", "")) == speaker_norm:
                    continue
                for key in specific_vote_keys(participant):
                    matched = False
                    if " " in key:
                        matched = bool(re.search(rf"\b{re.escape(key)}\b", vote_text))
                    else:
                        matched = key in vote_text_compact
                    if matched and len(key) > best_len:
                        best_index = index
                        best_len = len(key)

            if best_index is not None:
                scores[best_index] += 1
                last_vote_order.setdefault(best_index, len(last_vote_order))

        best_score = max(scores.values()) if scores else 0
        if best_score <= 0:
            return participants[0]

        tied = [index for index, score in scores.items() if score == best_score]
        if len(tied) == 1:
            return participants[tied[0]]

        # Deterministic tie-breaker: prefer the participant who reached the tied
        # score first, then fall back to the configured participant order.
        tied.sort(key=lambda index: (last_vote_order.get(index, 999), index))
        return participants[tied[0]]

    def _find_selected_model_targets(
        self, selected_model: Optional[str], participants: list[dict]
    ) -> list[dict]:
        if not selected_model:
            return []
        selected_model = str(selected_model)
        selected_norm = self._normalize_alias(selected_model)
        selected_compact = selected_norm.replace(" ", "")
        if not selected_norm or "conversation pipe" in selected_norm:
            return []

        matches = []
        for participant in participants:
            participant_values = [
                str(participant.get("model", "")),
                str(participant.get("alias", "")),
            ]
            for value in participant_values:
                value_norm = self._normalize_alias(value)
                if not value_norm:
                    continue
                if selected_norm == value_norm or selected_compact == value_norm.replace(" ", ""):
                    matches.append(participant)
                    break
        if matches:
            return matches

        return [
            {
                "model": selected_model,
                "alias": selected_model,
                "system_message": "",
            }
        ]


    def _find_direct_targets(self, text: str, participants: list[dict]) -> list[dict]:
        if not text:
            return []

        without_user_prefix = text
        if not re.match(r"^\s*@", without_user_prefix):
            without_user_prefix = re.sub(
                r"^\s*[A-Za-z0-9_. -]{1,30}:\s*", "", without_user_prefix, count=1
            )
        normalized = self._normalize_alias(without_user_prefix)
        compact = normalized.replace(" ", "")
        raw_lower = without_user_prefix.lower()

        def dedupe(items: list[dict]) -> list[dict]:
            deduped = []
            seen = set()
            for participant in items:
                marker = (participant.get("model"), participant.get("alias"))
                if marker not in seen:
                    seen.add(marker)
                    deduped.append(participant)
            return deduped

        def specific_keys(participant: dict) -> list[str]:
            keys = []
            for value in [participant.get("alias", ""), participant.get("model", "")]:
                key = self._normalize_alias(str(value))
                if not key:
                    continue
                for candidate in [key, key.replace(" ", "")]:
                    if candidate and candidate not in keys:
                        keys.append(candidate)
            return sorted(keys, key=len, reverse=True)

        # Exact @tag matching wins over fuzzy first-token matching. This keeps
        # @kimi-k2.6 from also selecting kimi-k2.7-code through the shared "kimi".
        at_tags = re.findall(r"@([A-Za-z0-9_.-]+)", raw_lower)
        exact_tag_targets = []
        for tag in at_tags:
            tag_norm = self._normalize_alias(tag)
            tag_compact = tag_norm.replace(" ", "")
            for participant in participants:
                keys = specific_keys(participant)
                if tag_norm in keys or tag_compact in keys:
                    exact_tag_targets.append(participant)
        if exact_tag_targets:
            return dedupe(exact_tag_targets)

        # Exact leading model/alias matching, before fuzzy "Kimi" matching.
        exact_start_matches = []
        for participant in participants:
            for key in specific_keys(participant):
                key_compact = key.replace(" ", "")
                if normalized == key or normalized.startswith(key + " "):
                    exact_start_matches.append((len(key), participant))
                    break
                if compact == key_compact or compact.startswith(key_compact):
                    exact_start_matches.append((len(key_compact), participant))
                    break
        if exact_start_matches:
            best_len = max(length for length, _ in exact_start_matches)
            return dedupe([participant for length, participant in exact_start_matches if length == best_len])

        # Exact "para <model>" matching, before fuzzy matching.
        exact_para_matches = []
        for participant in participants:
            for key in specific_keys(participant):
                key_compact = key.replace(" ", "")
                if re.search(rf"\bpara\s+{re.escape(key)}\b", normalized):
                    exact_para_matches.append((len(key), participant))
                    break
                if re.search(rf"\bpara{re.escape(key_compact)}\b", compact):
                    exact_para_matches.append((len(key_compact), participant))
                    break
        if exact_para_matches:
            best_len = max(length for length, _ in exact_para_matches)
            return dedupe([participant for length, participant in exact_para_matches if length == best_len])

        # Fallback: fuzzy aliases such as "Kimi, ..." may intentionally target
        # multiple similar participants.
        targets = []
        for participant in participants:
            for key in self._participant_match_keys(participant):
                key_compact = key.replace(" ", "")
                if re.search(rf"@{re.escape(key_compact)}\b", raw_lower):
                    targets.append(participant)
                    break

                starts_direct = normalized == key or normalized.startswith(key + " ")
                starts_compact = compact == key_compact or compact.startswith(key_compact)
                if starts_direct or starts_compact:
                    targets.append(participant)
                    break

                if re.search(rf"\bpara\s+{re.escape(key)}\b", normalized):
                    targets.append(participant)
                    break

        return dedupe(targets)

    def _has_leading_direct_target(self, text: str, participants: list[dict]) -> bool:
        if not text:
            return False

        candidate = text.strip()
        if not candidate.startswith("@"):
            candidate = re.sub(r"^\s*[A-Za-z0-9_. -]{1,30}:\s*", "", candidate, count=1).strip()

        at_match = re.match(r"^@([A-Za-z0-9_.-]+)\b", candidate)
        normalized = self._normalize_alias(candidate)
        compact = normalized.replace(" ", "")

        for participant in participants:
            values = [
                str(participant.get("alias", "") or ""),
                str(participant.get("model", "") or ""),
            ]
            for value in values:
                key = self._normalize_alias(value)
                if not key:
                    continue
                key_compact = key.replace(" ", "")
                if at_match:
                    tag_norm = self._normalize_alias(at_match.group(1))
                    if tag_norm == key or tag_norm.replace(" ", "") == key_compact:
                        return True
                if normalized == key or normalized.startswith(key + " "):
                    return True
                if compact == key_compact or compact.startswith(key_compact):
                    return True

        return False

    def _route_participants_for_message(
        self,
        text: str,
        participants: list[dict],
        rounds: int,
        use_manager: bool,
        selected_model: Optional[str] = None,
    ) -> tuple[list[dict], int, bool, str]:
        group_request = self._is_group_request(text)
        requested_rounds = self._requested_rounds(text)
        direct_targets = self._find_direct_targets(text, participants)
        selected_targets = self._find_selected_model_targets(selected_model, participants)
        leading_direct_target = self._has_leading_direct_target(text, participants)

        effective_rounds = requested_rounds or rounds
        effective_participants = participants
        effective_use_manager = use_manager
        route_reason = "group"

        if selected_targets:
            # Native @-chip selection from Open WebUI must always be an
            # explicit one-model turn, even if the message mentions debate,
            # round, everyone, or other group cues.
            effective_participants = selected_targets
            effective_rounds = 1
            effective_use_manager = False
            route_reason = "selected"
        elif direct_targets and (not group_request or leading_direct_target):
            effective_participants = direct_targets
            effective_rounds = 1
            effective_use_manager = False
            route_reason = "direct"
        elif requested_rounds:
            route_reason = "requested_rounds"

        return (
            effective_participants,
            max(1, effective_rounds),
            effective_use_manager,
            route_reason,
        )

    def _prefers_non_streaming(self, model: str) -> bool:
        normalized = self._normalize_alias(model)
        non_streaming_markers = [
            "kimi",
            "k2 7",
            "k27",
        ]
        return any(marker in normalized for marker in non_streaming_markers)

    def _build_config_js(self, default_valves: dict) -> str:
        # NOTE: This is a plain string, NOT an f-string.
        # We inject the defaults JSON via simple string replace to avoid Python
        # misinterpreting JS object literals like {id: ''} as f-string expressions
        # (since `id` is a Python builtin, f"{id: ''}" raises a TypeError).
        defaults_json = json.dumps(default_valves)
        js_code = r"""
return (function() {
  return new Promise(async (resolve) => {
    // Fetch models
    const defaults = __DEFAULTS_JSON__;
    let availableModels = Array.isArray(defaults.__available_models) ? defaults.__available_models : [];
    try {
      const token = localStorage.getItem('token') || localStorage.token || '';
      const res = await fetch('/api/models', {
        headers: token ? { 'Authorization': `Bearer ${token}` } : {},
        credentials: 'same-origin'
      });
      if (res.ok) {
        const json = await res.json();
        if (Array.isArray(json.data) && json.data.length > 0) {
          availableModels = json.data;
        }
      } else {
        console.warn('Failed to fetch models', res.status, await res.text());
      }
    } catch(e) {
      console.error('Failed to fetch models', e);
    }
    const MAX_PARTICIPANTS = 5;

    // Create UI overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed; inset: 0; z-index: 999999;
      background: rgba(0,0,0,0.6); backdrop-filter: blur(12px);
      display: flex; align-items: center; justify-content: center;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    `;

    const panel = document.createElement('div');
    panel.style.cssText = `
      background: rgba(20, 20, 25, 0.7); backdrop-filter: blur(20px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 24px; width: 95vw; max-width: 720px;
      max-height: 90vh; overflow-y: auto;
      box-shadow: 0 16px 48px rgba(0,0,0,0.4);
      display: flex; flex-direction: column; gap: 20px;
      color: #e2e8f0; scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent;
    `;
    overlay.appendChild(panel);

    const header = document.createElement('div');
    header.style.cssText = 'display: flex; flex-direction: column; gap: 4px;';
    const title = document.createElement('h2');
    title.textContent = '\u2728 Multi-Model Conversation';
    title.style.cssText = 'margin: 0; font-size: 20px; font-weight: 600; color: #fff; letter-spacing: -0.5px;';
    const subtitle = document.createElement('p');
    subtitle.textContent = 'Configure participants and conversation rules.';
    subtitle.style.cssText = 'margin: 0; font-size: 13px; color: #94a3b8;';
    header.appendChild(title);
    header.appendChild(subtitle);
    panel.appendChild(header);

    const form = document.createElement('div');
    form.style.cssText = 'display: flex; flex-direction: column; gap: 16px;';
    panel.appendChild(form);

    function createInputGrp(labelText, inputEl) {
      const grp = document.createElement('div');
      grp.style.cssText = 'display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 140px;';
      const lbl = document.createElement('label');
      lbl.textContent = labelText;
      lbl.style.cssText = 'font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;';
      grp.appendChild(lbl);
      inputEl.style.cssText = 'background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); color: #f8fafc; padding: 10px 14px; border-radius: 8px; font-size: 14px; outline: none; transition: all 0.2s; font-family: inherit; width: 100%; box-sizing: border-box;';
      inputEl.onfocus = () => { inputEl.style.borderColor = 'rgba(255,255,255,0.3)'; inputEl.style.background = 'rgba(0,0,0,0.4)'; };
      inputEl.onblur = () => { inputEl.style.borderColor = 'rgba(255,255,255,0.1)'; inputEl.style.background = 'rgba(0,0,0,0.3)'; };
      grp.appendChild(inputEl);
      return grp;
    }

    function createSelectGrp(labelText, options, defaultVal) {
      const sel = document.createElement('select');
      options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt.id; o.textContent = opt.name;
        sel.appendChild(o);
      });
      if (defaultVal) sel.value = defaultVal;
      return createInputGrp(labelText, sel);
    }

    const globalRow = document.createElement('div');
    globalRow.style.cssText = 'display: flex; gap: 16px; flex-wrap: wrap; background: rgba(0,0,0,0.25); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); align-items: flex-end;';

    const numPartsInp = document.createElement('input');
    numPartsInp.type = 'number'; numPartsInp.min = 1; numPartsInp.max = MAX_PARTICIPANTS; numPartsInp.value = defaults.NUM_PARTICIPANTS || 2;
    globalRow.appendChild(createInputGrp('Total Participants', numPartsInp));

    const managerDiv = document.createElement('div');
    managerDiv.style.cssText = 'display: flex; align-items: center; gap: 8px; height: 40px;';
    const managerChk = document.createElement('input');
    managerChk.type = 'checkbox'; managerChk.checked = defaults.UseGroupChatManager || false;
    managerChk.style.cssText = 'width: 18px; height: 18px; accent-color: #3b82f6; cursor: pointer;';
    const managerLbl = document.createElement('label');
    managerLbl.textContent = 'Auto-pilot (Use Manager)';
    managerLbl.style.cssText = 'font-size: 14px; font-weight: 500; color: #cbd5e1; cursor: pointer;';
    managerLbl.onclick = () => { managerChk.checked = !managerChk.checked; toggleManagerModel(); };
    managerDiv.appendChild(managerChk); managerDiv.appendChild(managerLbl);
    globalRow.appendChild(managerDiv);

    form.appendChild(globalRow);

    const modelOptions = [{id: '', name: 'Select Model...'}, ...availableModels.map(m => ({id: m.id, name: m.name}))];

    // Manager model selector (shown only when manager is enabled)
    const managerModelRow = document.createElement('div');
    managerModelRow.style.cssText = 'display: none; gap: 16px; flex-wrap: wrap; background: rgba(59,130,246,0.08); padding: 16px; border-radius: 12px; border: 1px solid rgba(59,130,246,0.2); align-items: flex-end;';
    const managerModelGrp = createSelectGrp('Manager Model (selects next speaker)', modelOptions, defaults.ManagerModel || '');
    const managerModelSel = managerModelGrp.querySelector('select');
    managerModelRow.appendChild(managerModelGrp);
    form.appendChild(managerModelRow);

    function toggleManagerModel() {
      managerModelRow.style.display = managerChk.checked ? 'flex' : 'none';
    }
    managerChk.onchange = toggleManagerModel;
    toggleManagerModel();

    const partsCont = document.createElement('div');
    partsCont.style.cssText = 'display: flex; flex-direction: column; gap: 12px;';
    form.appendChild(partsCont);

    const partUIs = [];

    function renderParticipants() {
      partsCont.innerHTML = '';
      partUIs.length = 0;
      let count = parseInt(numPartsInp.value);
      if (isNaN(count) || count < 1) count = 1;
      if (count > MAX_PARTICIPANTS) count = MAX_PARTICIPANTS;

      for (let i = 1; i <= count; i++) {
        const pbox = document.createElement('div');
        pbox.style.cssText = 'background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 16px; border-radius: 12px; display: flex; flex-direction: column; gap: 16px; transition: all 0.2s;';

        const ptit = document.createElement('div');
        ptit.textContent = `Participant ${i}`;
        ptit.style.cssText = 'font-size: 13px; font-weight: 600; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.8;';
        pbox.appendChild(ptit);

        const row = document.createElement('div');
        row.style.cssText = 'display: flex; gap: 16px; flex-wrap: wrap;';

        const defModel = defaults[`Participant${i}Model`] || '';
        const selGrp = createSelectGrp('Model', modelOptions, defModel);
        row.appendChild(selGrp);

        const defAlias = defaults[`Participant${i}Alias`] || '';
        const aliasInp = document.createElement('input');
        aliasInp.type = 'text'; aliasInp.value = defAlias; aliasInp.placeholder = 'e.g. Alice';
        row.appendChild(createInputGrp('Alias / Name', aliasInp));

        pbox.appendChild(row);

        const defSys = defaults[`Participant${i}SystemMessage`] || '';
        const sysInp = document.createElement('textarea');
        sysInp.value = defSys; sysInp.rows = 2;
        sysInp.placeholder = 'You are a helpful assistant...';
        sysInp.style.resize = 'vertical';
        pbox.appendChild(createInputGrp('System Prompt / Character Sheet', sysInp));

        partsCont.appendChild(pbox);
        partUIs.push({
          sel: selGrp.querySelector('select'),
          alias: aliasInp,
          sys: sysInp
        });
      }
    }
    numPartsInp.oninput = renderParticipants;
    renderParticipants();

    const actions = document.createElement('div');
    actions.style.cssText = 'display: flex; gap: 12px; margin-top: 8px; justify-content: flex-end;';

    const btnCancel = document.createElement('button');
    btnCancel.textContent = 'Skip / Use Defaults';
    btnCancel.style.cssText = 'background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.1); color: #fff; padding: 10px 20px; border-radius: 8px; font-size: 14px; cursor: pointer; transition: all 0.2s; font-weight: 500; font-family: inherit;';
    btnCancel.onmouseenter = () => { btnCancel.style.background = 'rgba(255,255,255,0.15)'; btnCancel.style.borderColor = 'rgba(255,255,255,0.2)'; };
    btnCancel.onmouseleave = () => { btnCancel.style.background = 'rgba(255,255,255,0.1)'; btnCancel.style.borderColor = 'rgba(255,255,255,0.1)'; };
    btnCancel.onclick = () => { cleanup(); resolve(null); };
    actions.appendChild(btnCancel);

    const btnStart = document.createElement('button');
    btnStart.textContent = 'Start Conversation \u2728';
    btnStart.style.cssText = 'background: linear-gradient(135deg, #3b82f6, #2563eb); border: none; color: #fff; padding: 10px 24px; border-radius: 8px; font-size: 14px; cursor: pointer; transition: all 0.2s; font-weight: 600; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); font-family: inherit; letter-spacing: 0.2px;';
    btnStart.onmouseenter = () => { btnStart.style.transform = 'translateY(-1px)'; btnStart.style.boxShadow = '0 6px 16px rgba(59, 130, 246, 0.5)'; };
    btnStart.onmouseleave = () => { btnStart.style.transform = 'none'; btnStart.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.4)'; };
    btnStart.onclick = () => {
      const config = {
        rounds: 1,
        use_manager: managerChk.checked,
        manager_model: managerChk.checked ? managerModelSel.value : '',
        participants: partUIs.map(ui => ({
          model: ui.sel.value,
          alias: ui.alias.value.trim() || (ui.sel.selectedIndex > 0 ? ui.sel.options[ui.sel.selectedIndex].text : 'Participant'),
          system_message: ui.sys.value.trim()
        })).filter(p => p.model !== '')
      };
      cleanup();
      resolve(config);
    };
    actions.appendChild(btnStart);
    panel.appendChild(actions);

    function cleanup() {
      if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
    }
    document.body.appendChild(overlay);
  });
})()
""".replace("__DEFAULTS_JSON__", defaults_json)
        return js_code

    def _parse_sse_events(self, buffer: str) -> tuple[list[dict], str, bool]:
        events = []
        done = False

        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in raw_event.splitlines():
                stripped = line.strip()
                if stripped.startswith("data:"):
                    data_lines.append(stripped[5:].lstrip())

            if not data_lines:
                continue

            payload = "\n".join(data_lines).strip()
            if not payload:
                continue

            if payload == "[DONE]":
                done = True
                break

            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    events.append(parsed)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed SSE payload: {payload[:200]}")

        return events, buffer, done

    def _extract_stream_events(self, event_payload: dict):
        choices = event_payload.get("choices", [])
        if not choices:
            return

        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta", {}) or {}

        reasoning_keys = ["reasoning", "reasoning_content", "thinking"]
        for reasoning_key in reasoning_keys:
            reasoning_text = delta.get(reasoning_key)
            if isinstance(reasoning_text, str) and reasoning_text:
                yield {"type": "reasoning", "text": reasoning_text}

        content = delta.get("content")
        if isinstance(content, str) and content:
            yield {"type": "content", "text": content}

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            yield {"type": "tool_calls", "data": tool_calls}

    def _replace_thinking_tags(self, text: str) -> str:
        """Replace raw thinking tags with <details> HTML the frontend understands."""
        text = THINK_OPEN_PATTERN.sub(
            '<details type="reasoning" done="false">\n<summary>Thinking…</summary>\n',
            text,
        )
        text = THINK_CLOSE_PATTERN.sub("\n</details>\n\n", text)
        return text

    async def get_streaming_completion(
        self, messages, model: str, valves, tools_specs=None
    ):
        try:
            form_data = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": valves.Temperature,
                "top_k": valves.Top_k,
                "top_p": valves.Top_p,
            }
            if tools_specs:
                form_data["tools"] = tools_specs
            response = await generate_raw_chat_completion(
                self.__request__,
                form_data,
                user=self.__user__,
                bypass_filter=True,
                bypass_system_prompt=True,
            )
            if not hasattr(response, "body_iterator"):
                raise ValueError("Response does not support streaming")

            sse_buffer = ""
            async for chunk in response.body_iterator:
                decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                sse_buffer += decoded
                events, sse_buffer, done = self._parse_sse_events(sse_buffer)
                for event_payload in events:
                    for event in self._extract_stream_events(event_payload):
                        yield event
                if done:
                    break
        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            yield {"type": "error", "text": f"\n\n**Error:** {e}\n\n"}

    async def get_completion(self, messages, model: str, valves) -> str:
        """Return a non-streaming completion, retrying with a minimal OpenAI payload.

        Some OpenAI-compatible providers reject extra sampling params such as top_k,
        and some Open WebUI adapters return response objects instead of plain dicts.
        """
        attempts = [
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": valves.Temperature,
                "top_p": valves.Top_p,
            },
            {
                "model": model,
                "messages": messages,
                "stream": False,
            },
        ]
        last_error = None
        last_shape = ""

        for form_data in attempts:
            try:
                response = await generate_chat_completions(
                    self.__request__,
                    form_data,
                    user=self.__user__,
                )

                if isinstance(response, dict):
                    if response.get("error"):
                        last_shape = f"error={response.get('error')}"
                        logger.warning(
                            f"Non-stream completion returned error for {model}: {last_shape}"
                        )
                        continue

                    choices = response.get("choices", [])
                    if choices and isinstance(choices[0], dict):
                        message = choices[0].get("message", {})
                        content = message.get("content", "") if isinstance(message, dict) else ""
                        if isinstance(content, str) and content.strip():
                            return content
                    last_shape = f"dict keys={list(response.keys())}"
                    logger.warning(
                        f"Non-stream completion returned no content for {model}: {last_shape}"
                    )
                    continue

                body = getattr(response, "body", None)
                if isinstance(body, bytes):
                    try:
                        parsed = json.loads(body.decode("utf-8"))
                        choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get("message", {})
                            content = message.get("content", "") if isinstance(message, dict) else ""
                            if isinstance(content, str) and content.strip():
                                return content
                    except Exception as parse_error:
                        last_error = parse_error

                last_shape = type(response).__name__
                logger.warning(
                    f"Non-stream completion returned unsupported response for {model}: {last_shape}"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"Non-stream completion attempt failed for {model}: {e}")

        if last_error:
            return f"_No pude obtener respuesta de {model}: {last_error}_"
        if last_shape:
            return f"_No pude obtener respuesta de {model}: respuesta vacia o no compatible ({last_shape})._"
        return f"_No pude obtener respuesta de {model}: respuesta vacia._"

    # ── Tool calling helpers ──────────────────────────────────────────

    async def _check_model_native_fc(self, model_id: str) -> bool:
        """Return True if the model is configured for native function calling."""
        model_info = await Models.get_model_by_id(model_id)
        if model_info and model_info.params:
            params = (
                model_info.params.model_dump()
                if hasattr(model_info.params, "model_dump")
                else {}
            )
            if params.get("function_calling") == "native":
                return True
        # Fall back to runtime MODELS state
        models = getattr(self.__request__.app.state, "MODELS", {})
        model = models.get(model_id, {})
        info = model.get("info", {})
        info_params = info.get("params", {})
        if isinstance(info_params, dict):
            return info_params.get("function_calling") == "native"
        if hasattr(info_params, "model_dump"):
            return info_params.model_dump().get("function_calling") == "native"
        return False

    async def _load_tools(self, tool_ids: list[str], extra_params: dict) -> dict:
        """Load tools using Open WebUI's get_tools(), returns tools_dict."""
        if not tool_ids:
            return {}
        return await get_tools(
            self.__request__,
            tool_ids,
            self.__user__,
            extra_params,
        )

    def _build_tool_call_details(
        self,
        call_id: str,
        name: str,
        arguments: str,
        done: bool = False,
        result=None,
        files=None,
        embeds=None,
    ) -> str:
        """Build <details type='tool_calls'> HTML matching Open WebUI's serialize_output()."""
        # arguments is already a JSON string, just escape for HTML attribute
        args_escaped = html_module.escape(arguments)
        if done:
            # result may be a string or other type
            result_text = (
                result
                if isinstance(result, str)
                else json.dumps(result or "", ensure_ascii=False)
            )
            result_escaped = html_module.escape(
                json.dumps(result_text, ensure_ascii=False)
            )
            files_escaped = html_module.escape(json.dumps(files)) if files else ""
            embeds_escaped = html_module.escape(json.dumps(embeds)) if embeds else ""
            return (
                f'<details type="tool_calls" done="true" id="{call_id}" '
                f'name="{name}" arguments="{args_escaped}" '
                f'result="{result_escaped}" files="{files_escaped}" '
                f'embeds="{embeds_escaped}">\n'
                f"<summary>Tool Executed</summary>\n</details>\n"
            )
        return (
            f'<details type="tool_calls" done="false" id="{call_id}" '
            f'name="{name}" arguments="{args_escaped}">\n'
            f"<summary>Executing...</summary>\n</details>\n"
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict],
        tools_dict: dict,
        metadata: dict,
        total_emitted: str,
    ) -> tuple[list[dict], str]:
        """Execute tool calls, emit <details> tags live. Returns (results, updated_total_emitted)."""
        results = []
        for tc in tool_calls:
            call_id = tc.get("id", str(uuid4()))
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            # Parse arguments
            params = {}
            try:
                params = ast.literal_eval(args_str)
            except Exception:
                try:
                    params = json.loads(args_str)
                except Exception:
                    logger.error(f"Failed to parse tool args for {name}: {args_str}")
                    results.append(
                        {
                            "tool_call_id": call_id,
                            "content": f"Error: malformed arguments for {name}",
                        }
                    )
                    continue

            # Emit "Executing..." tag — ensure newline before (middleware L369-370)
            executing_tag = self._build_tool_call_details(
                call_id, name, args_str, done=False
            )
            if total_emitted and not total_emitted.endswith("\n"):
                total_emitted += "\n"
            total_emitted += executing_tag
            await self.emit_replace(total_emitted)

            # Execute the tool
            tool_result = None
            tool = tools_dict.get(name)
            tool_type = tool.get("type", "") if tool else ""

            if tool and "callable" in tool:
                spec = tool.get("spec", {})
                allowed_params = spec.get("parameters", {}).get("properties", {}).keys()
                filtered_params = {
                    k: v for k, v in params.items() if k in allowed_params
                }
                try:
                    tool_result = await tool["callable"](**filtered_params)
                except Exception as e:
                    tool_result = str(e)
            else:
                tool_result = f"Error: tool '{name}' not found"

            # Process result using Open WebUI's process_tool_result
            tool_return = process_tool_result(
                self.__request__,
                name,
                tool_result,
                tool_type,
                False,
                metadata,
                self.__user__,
            )
            result_str = tool_return[0] if len(tool_return) > 0 else ""
            result_files = tool_return[1] if len(tool_return) > 1 else None
            result_embeds = tool_return[2] if len(tool_return) > 2 else None

            # Replace executing tag with completed tag
            total_emitted = total_emitted.replace(executing_tag, "")
            done_tag = self._build_tool_call_details(
                call_id,
                name,
                args_str,
                done=True,
                result=result_str,
                files=result_files,
                embeds=result_embeds,
            )
            if total_emitted and not total_emitted.endswith("\n"):
                total_emitted += "\n"
            total_emitted += done_tag
            await self.emit_replace(total_emitted)

            results.append(
                {
                    "tool_call_id": call_id,
                    "content": str(result_str) if result_str else "",
                }
            )
        return results, total_emitted

    async def _emit_accumulated_tool_files(self):
        """Emit a single combined chat:message:files event with all accumulated tool files."""
        if self._accumulated_tool_files and self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {
                    "type": "chat:message:files",
                    "data": {"files": list(self._accumulated_tool_files)},
                }
            )
            logger.debug(
                f"Emitted combined chat:message:files with {len(self._accumulated_tool_files)} file(s)"
            )
            # Clear after emitting so the next participant doesn't re-emit previous files
            self._accumulated_tool_files = []

    async def emit_message(self, message: str):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {"type": "message", "data": {"content": message}}
            )

    async def emit_replace(self, content: str):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {"type": "replace", "data": {"content": content}}
            )

    async def emit_status(self, level: str, message: str, done: bool):
        if self.__current_event_emitter__:
            await self.__current_event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )

    async def emit_model_title(self, model_name: str):
        await self.emit_message(f"\n\n### {model_name}\n\n")

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __event_call__: Callable[[Any], Awaitable[Any]] = None,
        __task__=None,
        __model__=None,
        __request__=None,
        __metadata__=None,
    ) -> str:
        if not hasattr(self, "_codex_pipe_lock"):
            self._codex_pipe_lock = asyncio.Lock()
        async with self._codex_pipe_lock:
            return await self._pipe_unlocked(
                body,
                __user__,
                __event_emitter__,
                __event_call__,
                __task__,
                __model__,
                __request__,
                __metadata__,
            )

    async def _pipe_unlocked(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __event_call__: Callable[[Any], Awaitable[Any]] = None,
        __task__=None,
        __model__=None,
        __request__=None,
        __metadata__=None,
    ) -> str:
        self.__current_event_emitter__ = __event_emitter__
        self.__user__ = await Users.get_user_by_id(__user__["id"])
        self.__model__ = __model__
        self.__request__ = __request__
        self.__metadata__ = __metadata__ or {}
        terminal_id = self.__metadata__.get("terminal_id") or body.get("terminal_id")

        valves = __user__.get("valves", self.UserValves())
        raw_history = await self._load_history_with_model_from_chat() or body.get("messages", [])

        conversation_history = []
        for msg in raw_history:
            cleaned_msg = msg.copy()
            if (
                cleaned_msg.get("role") == "assistant"
                and not cleaned_msg.get("content")
                and cleaned_msg.get("output")
            ):
                extracted_output_text = self._extract_text_from_output(
                    cleaned_msg.get("output")
                )
                if extracted_output_text:
                    cleaned_msg["content"] = extracted_output_text

            direct_model_id = str(cleaned_msg.get("model") or "")
            if (
                cleaned_msg.get("role") == "assistant"
                and direct_model_id
                and not self._is_conversation_pipe_model(direct_model_id)
                and not cleaned_msg.get("_speaker")
            ):
                cleaned_msg["_speaker"] = direct_model_id

            if "content" in cleaned_msg and isinstance(cleaned_msg["content"], str):
                cleaned_content = clean_thinking_tags(cleaned_msg["content"])

                if cleaned_msg.get("role") == "assistant" and not cleaned_msg.get(
                    "_speaker"
                ):
                    # Split concatenated multi-model messages back into individual speaker turns
                    # Optionally match the color circle emoji if present
                    parts = re.split(
                        r"(?:\n\n|^)### (?:(?:🔴|🔵|🟢|🟡|🟣|🟠|🟤|⚫|⚪)\s+)?(.+?)\n\n",
                        cleaned_content,
                    )

                    if len(parts) > 1:
                        if parts[0].strip():
                            conversation_history.append(
                                {"role": "assistant", "content": parts[0].strip()}
                            )

                        for i in range(1, len(parts), 2):
                            speaker_alias = parts[i].strip()
                            speaker_content = (
                                parts[i + 1].strip() if i + 1 < len(parts) else ""
                            )

                            if speaker_content:
                                conversation_history.append(
                                    {
                                        "role": "assistant",
                                        "content": speaker_content,
                                        "_speaker": speaker_alias,
                                    }
                                )
                        continue

                cleaned_msg["content"] = cleaned_content

            conversation_history.append(cleaned_msg)

        if not conversation_history:
            return "Error: No message history found."

        if __task__ and __task__ != TASKS.DEFAULT:
            # For tasks like title generation or summarization, use the first participant's context
            first_participant = (
                participants[0]
                if participants
                else {"model": self.__model__, "alias": "Assistant"}
            )
            target_model = first_participant.get("model", self.__model__)
            target_alias = first_participant.get("alias", "Assistant")

            # Adapt history for the target model to ensure strictly alternating turns
            adapted_task_history = self._adapt_history(
                conversation_history, target_alias
            )

            response = await generate_chat_completions(
                self.__request__,
                {
                    "model": target_model,
                    "messages": adapted_task_history,
                    "stream": False,
                },
                user=self.__user__,
            )
            return f"{name}: {response['choices'][0]['message']['content']}"

        # 1. Determine Configuration
        latest_user_text = self._latest_user_text(conversation_history)
        config = self._extract_config_from_metadata(body)
        loaded_config = None
        if not config:
            loaded_config = await self._load_config_from_chat_meta()
            config = loaded_config

        force_config_popup = self._should_open_config_popup(latest_user_text)
        if force_config_popup:
            config = None

        if not config and __event_call__:
            popup_seed_config = loaded_config or self._build_default_config_from_valves(valves)
            default_valves_dict = self._build_popup_defaults(valves, popup_seed_config)

            models_state = getattr(self.__request__.app.state, "MODELS", {}) or {}
            available_models = []
            for model_id, model_info in models_state.items():
                if not isinstance(model_info, dict):
                    continue
                pipeline = model_info.get("pipeline") or model_info.get("pipe") or {}
                if isinstance(pipeline, dict) and pipeline.get("type") == "filter":
                    continue
                if model_id == self.__model__:
                    continue
                available_models.append(
                    {
                        "id": model_id,
                        "name": model_info.get("name") or model_id,
                    }
                )
            default_valves_dict["__available_models"] = available_models

            js_code = self._build_config_js(default_valves_dict)
            await self.emit_status(
                "info", "Waiting for participant configuration...", False
            )
            try:
                result = await __event_call__(
                    {
                        "type": "execute",
                        "data": {
                            "code": js_code,
                        },
                    }
                )
                if isinstance(result, dict) and "participants" in result:
                    config = result
            except Exception as e:
                logger.error(f"Event call failed: {e}")

            if config:
                self._persist_config_to_metadata(body, config)

        if not config:
            default_config = self._build_default_config_from_valves(valves)
            if default_config.get("participants"):
                config = default_config

        config = self._sanitize_config(config, valves)
        self._persist_config_to_metadata(body, config)
        await self._persist_config_to_chat_meta(config)

        # Validate Participants
        participants = config.get("participants", [])
        if not participants:
            await self.emit_status("error", "No valid participants configured.", True)
            return "Error: No participants configured. Please set at least one participant."

        metadata_selected_model = self.__metadata__.get("selected_model")
        body_selected_model = body.get("selected_model")
        model_context = self.__model__
        if isinstance(model_context, dict):
            model_context = model_context.get("id") or model_context.get("model")
        selected_model = body_selected_model or metadata_selected_model or model_context
        selected_targets = self._find_selected_model_targets(selected_model, participants)
        if selected_targets:
            selected_participant = selected_targets[0]
            selected_model_norm = self._normalize_alias(
                selected_participant.get("model", "")
            )
            known_model_norms = {
                self._normalize_alias(participant.get("model", ""))
                for participant in participants
            }
            if (
                selected_model_norm
                and selected_model_norm not in known_model_norms
                and "conversation pipe" not in selected_model_norm
            ):
                participants.append(selected_participant)
                config["participants"] = participants
                self._persist_config_to_metadata(body, config)
                await self._persist_config_to_chat_meta(config)
                await self.emit_status(
                    "info",
                    f"Added {selected_participant.get('alias') or selected_participant.get('model')} "
                    "to this chat's participants.",
                    False,
                )

        latest_user_text = self._latest_user_text(conversation_history)
        inferred_targets = self._infer_participants_from_history(
            conversation_history, participants
        )
        current_text_targets = self._available_model_targets_from_text(
            latest_user_text, participants + inferred_targets
        )
        new_targets = inferred_targets + [
            target
            for target in current_text_targets
            if not self._participant_known(target, participants + inferred_targets)
        ]
        if new_targets:
            for target in new_targets:
                if not self._participant_known(target, participants):
                    participants.append(target)
            config["participants"] = participants
            self._persist_config_to_metadata(body, config)
            await self._persist_config_to_chat_meta(config)
            await self.emit_status(
                "info",
                "Added mentioned participant(s): "
                + ", ".join(target.get("alias") or target.get("model") for target in new_targets),
                False,
            )

        rounds = config.get("rounds", 3)
        use_manager = config.get("use_manager", False)
        manager_model = config.get("manager_model", "") or ""
        participants, rounds, use_manager, route_reason = self._route_participants_for_message(
            latest_user_text, participants, rounds, use_manager, selected_model
        )
        round_plan = self._round_plan(latest_user_text, rounds)
        # Keep vote counting scoped to the current user request. Otherwise, a
        # new problem in the same chat can inherit votes from a previous
        # debate and jump straight to the old winner/conclusion.
        phase_history_start = max(0, len(conversation_history) - 1)
        if route_reason == "direct":
            await self.emit_status(
                "info",
                "Directed turn: " + ", ".join(p["alias"] for p in participants),
                False,
            )
        elif route_reason == "requested_rounds":
            await self.emit_status(
                "info",
                f"Group conversation using {rounds} requested round(s).",
                False,
            )
        last_speaker = None

        # 2. Load tools (per-participant instead of global-only)
        # tool_ids come from metadata (which is popped from form_data by functions.py
        # before the pipe receives body, so we must read from __metadata__)
        global_tool_ids = self.__metadata__.get("tool_ids") or []
        if not global_tool_ids:
            # Fallback: check body in case it wasn't popped (e.g. direct API calls)
            global_tool_ids = body.get("tool_ids") or []

        logger.debug(
            f"[MultiModelTools] Global tool_ids from request: {global_tool_ids}"
        )

        # Proxy event emitter for tools: intercepts 'chat:message:files'
        self._accumulated_tool_files = []

        async def _tool_event_proxy(event):
            event_type = event.get("type", "")
            if event_type == "chat:message:files":
                files = event.get("data", {}).get("files", [])
                self._accumulated_tool_files.extend(files)
                logger.debug(
                    f"Captured {len(files)} file(s) from tool, "
                    f"total accumulated: {len(self._accumulated_tool_files)}"
                )
            else:
                if __event_emitter__:
                    await __event_emitter__(event)

        extra_params = {
            "__event_emitter__": _tool_event_proxy,
            "__event_call__": __event_call__,
            "__user__": __user__,
            "__request__": __request__,
            "__metadata__": self.__metadata__,
            "__chat_id__": self.__metadata__.get("chat_id"),
            "__message_id__": self.__metadata__.get("message_id"),
        }

        # Pre-load tools for each participant model
        participant_tools_map = {}
        models_state = getattr(self.__request__.app.state, "MODELS", {})

        # Load terminal tools once if terminal_id is present
        terminal_tools = {}
        if terminal_id:
            try:
                terminal_tools = await get_terminal_tools(
                    self.__request__, terminal_id, self.__user__, extra_params
                )
                logger.debug(
                    f"[MultiModelTools] Loaded {len(terminal_tools)} terminal tools"
                )
            except Exception as e:
                logger.error(f"[MultiModelTools] Failed to load terminal tools: {e}")

        for p in participants:
            p_model_id = p["model"]
            if p_model_id in participant_tools_map:
                continue  # already loaded for this model

            p_tools_dict = {}
            p_tools_specs = None

            # Fetch model info to get default tools (toolIds) and features
            model_info = models_state.get(p_model_id, {})
            model_db_info = await Models.get_model_by_id(p_model_id)

            logger.debug(
                f"[MultiModelTools] Loading tools for participant model: {p_model_id}"
            )

            default_tool_ids = []
            p_features = {}

            # 1. Start with global features from request body
            global_features = (
                self.__metadata__.get("features") or body.get("features") or {}
            )
            p_features.update(global_features)

            if model_db_info:
                meta = (
                    model_db_info.meta.model_dump()
                    if hasattr(model_db_info.meta, "model_dump")
                    else model_db_info.meta
                )
                params = (
                    model_db_info.params.model_dump()
                    if hasattr(model_db_info.params, "model_dump")
                    else model_db_info.params
                )

                if isinstance(meta, dict):
                    default_tool_ids.extend(meta.get("toolIds", []))
                    if "features" in meta and isinstance(meta["features"], dict):
                        p_features.update(meta["features"])
                    for f_id in meta.get("defaultFeatureIds", []):
                        p_features[f_id] = True

                if isinstance(params, dict):
                    default_tool_ids.extend(
                        params.get("toolIds", []) or params.get("tools", [])
                    )
                    if "features" in params and isinstance(params["features"], dict):
                        p_features.update(params["features"])
                    for f_id in params.get("defaultFeatureIds", []):
                        p_features[f_id] = True

            if model_info:
                info_meta = model_info.get("info", {}).get("meta", {})
                info_params = model_info.get("info", {}).get("params", {})

                default_tool_ids.extend(info_meta.get("toolIds", []))
                default_tool_ids.extend(
                    info_params.get("toolIds", []) or info_params.get("tools", [])
                )

                if isinstance(info_meta.get("features"), dict):
                    p_features.update(info_meta["features"])
                for f_id in info_meta.get("defaultFeatureIds", []):
                    p_features[f_id] = True

                if isinstance(info_params.get("features"), dict):
                    p_features.update(info_params["features"])
                for f_id in info_params.get("defaultFeatureIds", []):
                    p_features[f_id] = True

            # Clean and deduplicate tool IDs (handling cases where UI might inadvertently store dicts)
            clean_default_ids = []
            for t in default_tool_ids:
                if isinstance(t, str) and t.strip():
                    clean_default_ids.append(t.strip())
                elif isinstance(t, dict) and "id" in t:
                    clean_default_ids.append(str(t["id"]))

            # Combine global tools and model's default tools uniquely
            combined_tool_ids = list(set(global_tool_ids + clean_default_ids))

            # Load user-imported tools
            if combined_tool_ids:
                try:
                    p_tools_dict = await self._load_tools(
                        combined_tool_ids, extra_params
                    )
                except Exception as e:
                    logger.error(
                        f"[MultiModelTools] Failed to load imported tools for {p_model_id}: {e}"
                    )

            # Load built-in tools (web search, knowledge, etc.) based on the specific model.
            # Normalize legacy knowledge items so query_knowledge_files recognizes them
            # (it expects {"type": ..., "id": ...} but DB may have {"collection_name": ...}).
            p_model_info = model_info
            try:
                raw_knowledge = (
                    model_info.get("info", {}).get("meta", {}).get("knowledge", [])
                    or []
                )

                # Fallback: if MODELS state has no knowledge, try reading from DB directly
                if not raw_knowledge and model_db_info and model_db_info.meta:
                    m = (
                        model_db_info.meta.model_dump()
                        if hasattr(model_db_info.meta, "model_dump")
                        else model_db_info.meta
                    )
                    if isinstance(m, dict):
                        raw_knowledge = m.get("knowledge", []) or []

                if raw_knowledge:
                    normalized = []
                    needs_normalization = False
                    for item in raw_knowledge:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") and item.get("id"):
                            normalized.append(item)
                        elif item.get("collection_name"):
                            normalized.append(
                                {
                                    "type": "collection",
                                    "id": item["collection_name"],
                                    "name": item.get("name", ""),
                                }
                            )
                            needs_normalization = True
                        else:
                            normalized.append(item)

                    # Create a shallow copy to avoid mutating shared MODELS state
                    p_model_info = {**model_info}
                    p_model_info["info"] = {**p_model_info.get("info", {})}
                    p_model_info["info"]["meta"] = {
                        **p_model_info["info"].get("meta", {})
                    }
                    p_model_info["info"]["meta"]["knowledge"] = normalized
                    if needs_normalization:
                        logger.debug(
                            f"[MultiModelTools] Normalized legacy knowledge items for {p_model_id}"
                        )
            except Exception as e:
                logger.error(
                    f"[MultiModelTools] Knowledge normalization failed for {p_model_id}: {e}"
                )
                p_model_info = model_info  # fall back to original

            logger.debug(f"[MultiModelTools] Features for {p_model_id}: {p_features}")
            logger.debug(
                f"[MultiModelTools] ENABLE_IMAGE_GENERATION={getattr(getattr(self.__request__.app.state, 'config', None), 'ENABLE_IMAGE_GENERATION', 'NOT_SET')}"
            )

            try:
                builtin_tools = await get_builtin_tools(
                    self.__request__,
                    extra_params,
                    features=p_features,
                    model=p_model_info,
                )
                if builtin_tools:
                    p_tools_dict.update(builtin_tools)
                    logger.debug(
                        f"[MultiModelTools] Builtin tools loaded for {p_model_id}: {list(builtin_tools.keys())}"
                    )
                else:
                    logger.debug(
                        f"[MultiModelTools] No builtin tools returned for {p_model_id}"
                    )
            except Exception as e:
                logger.error(
                    f"[MultiModelTools] Failed to load built-in tools for {p_model_id}: {e}"
                )

            # Add terminal tools if present
            if terminal_tools:
                p_tools_dict.update(terminal_tools)
                logger.debug(f"[MultiModelTools] Terminal tools added for {p_model_id}")

            # Get native function calling flag
            native_fc = await self._check_model_native_fc(p_model_id)

            # Get system message
            model_system_message = ""
            if model_db_info and model_db_info.params:
                p_params = (
                    model_db_info.params.model_dump()
                    if hasattr(model_db_info.params, "model_dump")
                    else model_db_info.params
                )
                if isinstance(p_params, dict):
                    model_system_message = p_params.get("system", "") or ""

            # Build OpenAI-format tool specs
            if p_tools_dict:
                p_tools_specs = [
                    {"type": "function", "function": t.get("spec", {})}
                    for t in p_tools_dict.values()
                ]

            participant_tools_map[p_model_id] = {
                "dict": p_tools_dict,
                "specs": p_tools_specs,
                "native_fc": native_fc,
                "system_message": model_system_message,
            }

        MAX_TOOL_CALL_RETRIES = 5
        MAX_MALFORMED_RETRIES = 2  # Silent retries for malformed tool calls

        # 3. Run Conversation Rounds
        # total_emitted tracks ALL content sent to the frontend across
        # all participants so that replace events preserve prior output
        #
        # One "round" = len(participants) turns.
        # Without manager: each participant speaks once per round in order.
        # With manager: the manager picks who speaks each turn (no consecutive repeats).
        # Total turns = len(participants) * rounds.
        total_emitted = ""
        turns_per_round = len(participants)

        # Resolve the manager model once (fall back to first participant's model)
        if use_manager:
            resolved_manager_model = (
                manager_model or valves.ManagerModel or participants[0]["model"]
            )
            if not resolved_manager_model:
                await self.emit_status(
                    "error",
                    "No manager model configured. Please set a Manager Model.",
                    True,
                )
                return "Error: No manager model configured."
            logger.debug(
                f"[MultiModelTools] Using manager model: {resolved_manager_model}"
            )

        for round_num in range(rounds):
            round_phase = (
                round_plan[round_num] if round_num < len(round_plan) else "debate"
            )
            # Vote and conclusion phases are deterministic:
            # - vote: each configured participant votes exactly once;
            # - conclusion: only the selected winner speaks once.
            # Debate can still use the optional manager to pick speakers.
            num_turns = 1 if round_phase in ["vote", "conclusion"] else (turns_per_round if use_manager else 1)

            for turn in range(num_turns):
                if round_phase == "vote":
                    participants_to_run = participants
                elif round_phase == "conclusion":
                    participants_to_run = [
                        self._select_conclusion_participant(
                            conversation_history[phase_history_start:], participants
                        )
                    ]
                elif use_manager:
                    participant_aliases = [p["alias"] for p in participants]
                    # Build a clean conversation dump for the manager:
                    # only user messages and final assistant text (with speaker labels),
                    # no tool calls, files, embeds, or internal metadata.
                    history_lines = []
                    for msg in conversation_history:
                        speaker = msg.get("_speaker")
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if not content or not content.strip():
                            continue
                        if speaker:
                            # Participant response — label with alias
                            history_lines.append(f"{speaker}: {content.strip()}")
                        elif role == "user":
                            history_lines.append(f"User: {content.strip()}")
                        elif role == "system":
                            continue  # skip system prompts
                        else:
                            history_lines.append(f"{role}: {content.strip()}")
                    history_str = "\n".join(history_lines)
                    manager_prompt = valves.ManagerSelectionPrompt.format(
                        history=history_str,
                        last_speaker=last_speaker or "None",
                        participant_list=", ".join(participant_aliases),
                    )
                    manager_messages = [
                        {"role": "system", "content": valves.ManagerSystemMessage},
                        {"role": "user", "content": manager_prompt},
                    ]

                    await self.emit_status(
                        "info",
                        f"Round {round_num + 1}/{rounds} — Turn {turn + 1}/{turns_per_round}: Manager selecting next speaker...",
                        False,
                    )

                    try:
                        manager_response = await generate_chat_completions(
                            self.__request__,
                            {
                                "model": resolved_manager_model,
                                "messages": manager_messages,
                                "stream": False,
                            },
                            user=self.__user__,
                        )
                        selected_alias = manager_response["choices"][0]["message"][
                            "content"
                        ].strip()
                    except Exception as e:
                        logger.error(f"Manager failed to select: {e}")
                        selected_alias = ""

                    normalized_selected_alias = self._normalize_alias(selected_alias)
                    selected_participant = next(
                        (
                            p
                            for p in participants
                            if self._normalize_alias(p["alias"])
                            == normalized_selected_alias
                        ),
                        None,
                    )

                    if not selected_participant or self._normalize_alias(
                        selected_participant["alias"]
                    ) == self._normalize_alias(last_speaker or ""):
                        await self.emit_status(
                            "info",
                            "Manager selection was invalid/repeated. Falling back.",
                            False,
                        )
                        last_speaker_index = next(
                            (
                                i
                                for i, p in enumerate(participants)
                                if p["alias"] == last_speaker
                            ),
                            -1,
                        )
                        fallback_index = (last_speaker_index + 1) % len(participants)
                        selected_participant = participants[fallback_index]

                    participants_to_run = [selected_participant]
                    last_speaker = selected_participant["alias"]
                else:
                    participants_to_run = participants

                for participant in participants_to_run:
                    model = participant["model"]
                    alias = participant["alias"]

                    # Determine if this model can use tools
                    participant_tools_specs = None
                    p_tools = participant_tools_map.get(model, {})
                    tools_dict = p_tools.get("dict", {})
                    tools_specs = p_tools.get("specs", None)
                    native_fc = p_tools.get("native_fc", False)
                    model_system_message = p_tools.get("system_message", "")

                    if tools_dict and tools_specs:
                        if native_fc:
                            participant_tools_specs = tools_specs
                            logger.debug(
                                f"[MultiModelTools] Sending {len(tools_specs)} tool specs to {alias} ({model})"
                            )
                        else:
                            logger.warning(
                                f"Model {model} does not have native function calling enabled. "
                                f"Tools will be skipped for {alias}."
                            )

                    # Build system prompt: model's system message + participant config + conversation flow
                    system_parts = []
                    if model_system_message.strip():
                        system_parts.append(model_system_message.strip())
                    if participant["system_message"].strip():
                        system_parts.append(participant["system_message"].strip())
                    system_parts.append(
                        f"{valves.AllParticipantsApendedMessage} {alias}\n\n"
                        "This is a moderated multi-participant conversation. The human user David is the moderator. "
                        "Your primary job is to answer David's latest message directly and helpfully. "
                        "Messages from other AI participants appear as user messages labeled with their name; use them as context, "
                        "but do not let the AI participants continue a side conversation unless David explicitly asks for debate or dialogue between models. "
                        "Do not claim messages written by other participants as your own. "
                        "Respond naturally in character without prefixing your name."
                    )

                    # If this participant has tools, add explicit tool-use guidance
                    if participant_tools_specs:
                        tool_names = [
                            t["function"]["name"]
                            for t in participant_tools_specs
                            if "function" in t
                        ]
                        system_parts.append(
                            "TOOL USE INSTRUCTIONS:\n"
                            f"You have access to the following tools: {', '.join(tool_names)}.\n"
                            "When the user's request can be fulfilled by a tool, call the appropriate tool.\n"
                            "CRITICAL: After every tool call, once you receive the tool results, "
                            "you MUST produce a visible text response summarizing or presenting the result to the user. "
                            "Never stop at just the tool call — always follow up with a written reply."
                        )

                    system_prompt = "\n\n".join(system_parts)

                    # Build messages with role-based separation and strict turn alternation:
                    messages = [
                        {"role": "system", "content": system_prompt}
                    ] + self._adapt_history(conversation_history, alias)

                    if rounds > 1:
                        round_number = round_num + 1
                        round_note = self._phase_instruction(
                            round_phase, round_number, rounds, alias
                        )
                        messages.append({"role": "user", "content": round_note})

                    await self.emit_status(
                        "info", f"Getting response from: {alias} ({model})...", False
                    )

                    try:
                        p_idx = participants.index(participant) % len(SPEAKER_COLORS)
                        color = SPEAKER_COLORS[p_idx]
                        title_text = f"\n\n### {color} {alias}\n\n"
                        total_emitted += title_text
                        await self.emit_replace(total_emitted)

                        # Emit warning if tools configured but model lacks native FC
                        if tools_dict and not participant_tools_specs:
                            warn_tag = (
                                '<details type="tool_calls" done="true" id="warn" '
                                f'name="⚠️ Tools Unavailable" '
                                f'arguments="&quot;&quot;" '
                                f'result="&quot;Model {html_module.escape(model)} does not have '
                                f'native function calling enabled. Tools skipped for {html_module.escape(alias)}.&quot;" '
                                f'files="" embeds="">\n'
                                f"<summary>Tools Unavailable</summary>\n</details>\n"
                            )
                            total_emitted += warn_tag
                            await self.emit_replace(total_emitted)

                        full_response = ""
                        reasoning_buffer = ""
                        reasoning_start_time = None
                        accumulated_tool_calls = []
                        metadata = body.get("metadata", {})

                        # ── Streaming + tool call accumulation helper ────
                        async def _stream_and_accumulate(stream_messages, tc_specs):
                            nonlocal full_response, reasoning_buffer
                            nonlocal reasoning_start_time, total_emitted
                            nonlocal accumulated_tool_calls

                            accumulated_tool_calls = []
                            last_emit_time = 0.0
                            emit_interval = 0.1

                            async def throttled_emit_replace(html_content, force=False):
                                nonlocal last_emit_time
                                now = time.monotonic()
                                if force or (now - last_emit_time >= emit_interval):
                                    await self.emit_replace(html_content)
                                    last_emit_time = now

                            async for event in self.get_streaming_completion(
                                stream_messages,
                                model=model,
                                valves=valves,
                                tools_specs=tc_specs,
                            ):
                                event_type = event.get("type")
                                if event_type == "error":
                                    error_text = event.get("text", "")
                                    if "does not support streaming" in error_text.lower():
                                        logger.warning(
                                            f"Streaming unsupported by {alias} ({model}); falling back to non-streaming completion."
                                        )
                                        continue
                                    total_emitted += error_text
                                    await throttled_emit_replace(
                                        total_emitted, force=True
                                    )
                                    continue

                                if event_type == "tool_calls":
                                    # Accumulate tool call deltas (same as middleware.py)
                                    for delta_tc in event.get("data", []):
                                        tc_index = delta_tc.get("index")
                                        if tc_index is not None:
                                            existing = None
                                            for atc in accumulated_tool_calls:
                                                if atc.get("index") == tc_index:
                                                    existing = atc
                                                    break
                                            if existing is None:
                                                delta_tc.setdefault("function", {})
                                                delta_tc["function"].setdefault(
                                                    "name", ""
                                                )
                                                delta_tc["function"].setdefault(
                                                    "arguments", ""
                                                )
                                                accumulated_tool_calls.append(delta_tc)
                                            else:
                                                dn = delta_tc.get("function", {}).get(
                                                    "name"
                                                )
                                                da = delta_tc.get("function", {}).get(
                                                    "arguments"
                                                )
                                                if dn:
                                                    existing["function"]["name"] += dn
                                                if da:
                                                    existing["function"][
                                                        "arguments"
                                                    ] += da
                                    continue

                                if event_type == "reasoning":
                                    reasoning_piece = event.get("text", "")
                                    if reasoning_piece:
                                        if reasoning_start_time is None:
                                            reasoning_start_time = time.monotonic()
                                        reasoning_buffer += reasoning_piece
                                        # Format with blockquote prefix (> ) like middleware
                                        display = "\n".join(
                                            (
                                                f"> {line}"
                                                if not line.startswith(">")
                                                else line
                                            )
                                            for line in reasoning_buffer.splitlines()
                                        )
                                        await throttled_emit_replace(
                                            total_emitted
                                            + '<details type="reasoning" done="false">\n'
                                            + "<summary>Thinking...</summary>\n"
                                            + display
                                            + "\n</details>\n\n"
                                        )
                                    continue

                                # Finalize thinking block when transitioning
                                if reasoning_buffer:
                                    reasoning_duration = (
                                        round(time.monotonic() - reasoning_start_time)
                                        if reasoning_start_time
                                        else 1
                                    )
                                    display = "\n".join(
                                        (
                                            f"> {line}"
                                            if not line.startswith(">")
                                            else line
                                        )
                                        for line in reasoning_buffer.splitlines()
                                    )
                                    total_emitted += (
                                        f'<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                                        f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                                        + display
                                        + "\n</details>\n\n"
                                    )
                                    reasoning_buffer = ""
                                    await throttled_emit_replace(
                                        total_emitted, force=True
                                    )

                                if event_type == "content":
                                    chunk_text = event.get("text", "")
                                    if not chunk_text:
                                        continue
                                    full_response += chunk_text
                                    formatted_chunk = self._replace_thinking_tags(
                                        chunk_text
                                    )
                                    total_emitted += formatted_chunk
                                    await self.emit_message(formatted_chunk)
                                    continue

                            # Flush reasoning if stream ended during thinking
                            if reasoning_buffer:
                                reasoning_duration = (
                                    round(time.monotonic() - reasoning_start_time)
                                    if reasoning_start_time
                                    else 1
                                )
                                display = "\n".join(
                                    f"> {line}" if not line.startswith(">") else line
                                    for line in reasoning_buffer.splitlines()
                                )
                                total_emitted += (
                                    f'<details type="reasoning" done="true" duration="{reasoning_duration}">\n'
                                    f"<summary>Thought for {reasoning_duration} seconds</summary>\n"
                                    + display
                                    + "\n</details>\n\n"
                                )
                                reasoning_buffer = ""

                            # Final flush at the end of the stream
                            await throttled_emit_replace(total_emitted, force=True)

                        # ── Initial model call ─────────────────────────
                        tool_interaction_messages = []
                        if self._prefers_non_streaming(model) and not participant_tools_specs:
                            await self.emit_status(
                                "info",
                                f"Using non-streaming completion for: {alias} ({model})...",
                                False,
                            )
                            full_response = await self.get_completion(
                                messages, model=model, valves=valves
                            )
                            if full_response.strip():
                                total_emitted += full_response
                                await self.emit_replace(total_emitted)
                        else:
                            await _stream_and_accumulate(messages, participant_tools_specs)

                        # ── Tool call execution + re-prompt loop ────────
                        if accumulated_tool_calls and participant_tools_specs:
                            tool_call_retries = 0
                            malformed_retries = 0
                            current_messages = list(messages)

                            while (
                                accumulated_tool_calls
                                and tool_call_retries < MAX_TOOL_CALL_RETRIES
                            ):
                                tool_call_retries += 1

                                # ── Silent retry for malformed tool calls ───
                                malformed = []
                                valid = []
                                for tc in accumulated_tool_calls:
                                    func = tc.get("function", {})
                                    tc_name = func.get("name", "").strip()
                                    tc_args = func.get("arguments", "{}")

                                    # Check: name must exist and be a known tool
                                    if not tc_name or tc_name not in tools_dict:
                                        malformed.append(
                                            f"Unknown tool '{tc_name}'"
                                            if tc_name
                                            else "Tool call with empty name"
                                        )
                                        continue

                                    # Check: arguments must be parseable
                                    try:
                                        ast.literal_eval(tc_args)
                                    except Exception:
                                        try:
                                            json.loads(tc_args)
                                        except Exception:
                                            malformed.append(
                                                f"Malformed arguments for '{tc_name}': {tc_args[:100]}"
                                            )
                                            continue

                                    valid.append(tc)

                                if (
                                    malformed
                                    and malformed_retries < MAX_MALFORMED_RETRIES
                                ):
                                    malformed_retries += 1
                                    logger.warning(
                                        f"[MultiModelTools] Silent retry {malformed_retries}/{MAX_MALFORMED_RETRIES} "
                                        f"for {alias}: {'; '.join(malformed)}"
                                    )

                                    # Build correction messages without showing anything to user
                                    available_tools = list(tools_dict.keys())
                                    correction_msg = {
                                        "role": "user",
                                        "content": (
                                            f"Your previous tool call was malformed: {'; '.join(malformed)}. "
                                            f"Available tools are: {', '.join(available_tools)}. "
                                            "Please try again with a valid tool name and properly formatted JSON arguments. "
                                            "After the tool call, always provide a text response."
                                        ),
                                    }
                                    current_messages.append(correction_msg)

                                    # Reset and re-call model
                                    full_response = ""
                                    accumulated_tool_calls = []
                                    await _stream_and_accumulate(
                                        current_messages, participant_tools_specs
                                    )
                                    continue

                                # If some valid and some malformed (past retry limit),
                                # just execute the valid ones
                                if malformed and valid:
                                    logger.warning(
                                        f"[MultiModelTools] Skipping {len(malformed)} malformed tool call(s), "
                                        f"executing {len(valid)} valid one(s) for {alias}"
                                    )
                                    accumulated_tool_calls = valid
                                elif malformed and not valid:
                                    logger.warning(
                                        f"[MultiModelTools] All tool calls malformed for {alias}, "
                                        "skipping tool execution"
                                    )
                                    accumulated_tool_calls = []
                                    break

                                # Execute the accumulated tool calls
                                results, total_emitted = await self._execute_tool_calls(
                                    accumulated_tool_calls,
                                    tools_dict,
                                    metadata,
                                    total_emitted,
                                )
                                # Emit all accumulated files as single combined events
                                await self._emit_accumulated_tool_files()

                                # Build re-prompt messages (OpenAI format):
                                # 1. Assistant message with tool_calls
                                assistant_tc_msg = {
                                    "role": "assistant",
                                    "content": full_response or None,
                                    "tool_calls": [
                                        {
                                            "id": tc.get("id", str(uuid4())),
                                            "type": "function",
                                            "function": tc.get("function", {}),
                                        }
                                        for tc in accumulated_tool_calls
                                    ],
                                }
                                current_messages.append(assistant_tc_msg)
                                tool_interaction_messages.append(assistant_tc_msg)

                                # 2. Tool result messages
                                for result in results:
                                    tool_result_msg = {
                                        "role": "tool",
                                        "tool_call_id": result["tool_call_id"],
                                        "content": result.get("content", ""),
                                    }
                                    current_messages.append(tool_result_msg)
                                    tool_interaction_messages.append(tool_result_msg)

                                # Reset for next streaming round
                                full_response = ""

                                # On last retry, don't pass tools so model is forced to produce text
                                reprompt_tools = (
                                    participant_tools_specs
                                    if tool_call_retries < MAX_TOOL_CALL_RETRIES
                                    else None
                                )

                                # Re-call model with tool results
                                await _stream_and_accumulate(
                                    current_messages, reprompt_tools
                                )

                        # ── Fallback for empty response ─────────────────
                        if not full_response.strip():
                            await self.emit_status(
                                "info",
                                f"Empty stream from {alias} ({model}). Retrying once (non-stream).",
                                False,
                            )
                            # Use current_messages (includes tool results) if available,
                            # otherwise fall back to original messages
                            fallback_msgs = (
                                current_messages
                                if tool_interaction_messages
                                else messages
                            )
                            fallback_response = await self.get_completion(
                                fallback_msgs, model=model, valves=valves
                            )
                            if fallback_response.strip():
                                full_response = fallback_response
                                total_emitted += fallback_response
                                await self.emit_replace(total_emitted)
                            else:
                                no_response_text = (
                                    f"_No pude obtener una respuesta visible de {alias} ({model})._"
                                )
                                full_response = no_response_text
                                total_emitted += no_response_text
                                await self.emit_replace(total_emitted)
                                await self.emit_status(
                                    "warning",
                                    f"No response produced by {alias} ({model}).",
                                    False,
                                )

                        cleaned_response = clean_thinking_tags(full_response)
                        if cleaned_response.strip():
                            conversation_history.append(
                                {
                                    "role": "assistant",
                                    "content": cleaned_response.strip(),
                                    "_speaker": alias,
                                    "_tool_messages": tool_interaction_messages,
                                }
                            )

                    except Exception as e:
                        error_message = (
                            f"Error getting response from {alias} ({model}): {e}"
                        )
                        await self.emit_status("error", error_message, True)
                        await self.emit_message(f"\n\n**ERROR**: {error_message}\n\n")

        await self.emit_status("success", "Conversation round completed.", True)

        # When returning from pipe directly, we yield an empty string because we used emit_message
        return ""
