import re
from typing import Dict


class TaskClassifierTool:
    name = "task_classifier"
    description = "Classifies a user request into supported task types"

    # Keyword sets
    TECHNICAL_KEYWORDS = {
        "explain", "compare", "outline", "describe",
        "architecture", "design", "approach", "tradeoff",
        "how does", "how to"
    }

    DATA_KEYWORDS = {
        "normalize", "transform", "extract",
        "parse", "convert", "table", "json", "csv"
    }

    EXECUTION_KEYWORDS = {
        "run", "execute", "deploy", "ssh", "curl",
        "token", "apikey", "password"
    }

    UNSAFE_PATTERNS = [
        r"api[_\- ]?key",
        r"secret",
        r"password",
        r"-----BEGIN (RSA|EC|DSA) PRIVATE KEY-----",
        r"rm\s+-rf",
        r"\bexec\b",
        r"\beval\b",
        r"\bsubprocess\b",
    ]

    def invoke(self, input_data: Dict) -> Dict:
        prompt = input_data.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt is required for task classification")

        text = prompt.lower()

        # Hard safety block
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, text):
                return {
                    "output": {
                        "task_type": "rejected",
                        "confidence": 1.0,
                        "reason": "unsafe_content_detected"
                    },
                    "state_updates": {
                        "task_type": "rejected"
                    }
                }

        tech_hits = sum(k in text for k in self.TECHNICAL_KEYWORDS)
        data_hits = sum(k in text for k in self.DATA_KEYWORDS)
        exec_hits = sum(k in text for k in self.EXECUTION_KEYWORDS)

        if exec_hits > 0:
            return {
                "output": {
                    "task_type": "rejected",
                    "confidence": 0.9,
                    "reason": "execution_request_detected"
                },
                "state_updates": {
                    "task_type": "rejected"
                }
            }

        if tech_hits >= 1:
            confidence = min(0.6 + 0.1 * tech_hits, 0.95)
            return {
                "output": {
                    "task_type": "technical_brief",
                    "confidence": confidence
                },
                "state_updates": {
                    "task_type": "technical_brief"
                }
            }

        if data_hits >= 1:
            confidence = min(0.6 + 0.1 * data_hits, 0.95)
            return {
                "output": {
                    "task_type": "data_transformation",
                    "confidence": confidence
                },
                "state_updates": {
                    "task_type": "data_transformation"
                }
            }

        return {
            "output": {
                "task_type": "unknown",
                "confidence": 0.3
            },
            "state_updates": {
                "task_type": "unknown"
            }
        }

class TopicExtractorTool:
    name = "topic_extractor"
    description = "Extracts the core technical topic from a user prompt"

    def invoke(self, input_data: dict) -> dict:
        prompt = input_data.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Prompt is required")

        # Simple heuristic: first sentence or clause
        topic = prompt.split(".")[0][:120]

        if len(topic) < 5:
            raise ValueError("Extracted topic too short")

        return {
            "output": topic,
            "state_updates": {
                "topic": topic
            }
        }

    