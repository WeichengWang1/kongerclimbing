from __future__ import annotations

# Coaching text templates keyed by (issue code, severity).
# Each entry: {"message": ..., "recommendation": ...}
# When confidence < _HEDGE_THRESHOLD the message is prefixed with "Possible: "
# to signal that the detection is uncertain.

_HEDGE_THRESHOLD = 0.30

_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "over_pulling_with_arms": {
        "low": {
            "message": "You may be using more arm strength than necessary for this section.",
            "recommendation": (
                "Try engaging your leg drive earlier — "
                "push your hips toward the wall as you reach."
            ),
        },
        "medium": {
            "message": (
                "You are pulling hard with your arms while "
                "your hips and legs contribute little."
            ),
            "recommendation": (
                "Drive from your legs first: push your hips up, "
                "then use your arms to guide the movement."
            ),
        },
        "high": {
            "message": (
                "Your arms are doing nearly all the work "
                "while your lower body is passive."
            ),
            "recommendation": (
                "Reset and commit to using your legs. Plant your feet, engage your core, "
                "and push your body upward before pulling."
            ),
        },
    },
    "unstable_center_of_mass": {
        "low": {
            "message": "Your center of mass shows minor oscillation during this section.",
            "recommendation": (
                "Aim for a single smooth arc of movement — "
                "pause briefly at each hold to stabilize before transitioning."
            ),
        },
        "medium": {
            "message": (
                "Your center of mass is moving erratically, "
                "reversing direction multiple times."
            ),
            "recommendation": (
                "Move your center of mass in one deliberate arc. "
                "Identify your next hold before committing to the move."
            ),
        },
        "high": {
            "message": (
                "Your center of mass is oscillating significantly, "
                "wasting energy and affecting balance."
            ),
            "recommendation": (
                "Break the move into smaller steps: stabilize fully at each hold "
                "before committing to the next. Avoid rushing transitions."
            ),
        },
    },
    "poor_foot_engagement": {
        "low": {
            "message": (
                "Your feet are shifting slightly on the holds, "
                "possibly indicating insecure placement."
            ),
            "recommendation": (
                "Consciously press your toes into the hold and shift your weight "
                "over your support foot before moving."
            ),
        },
        "medium": {
            "message": (
                "Your feet seem unstable on the holds and your weight "
                "is not shifting over the support foot."
            ),
            "recommendation": (
                "Keep your feet quiet: place them deliberately, trust the hold, "
                "and shift your hips directly over the support foot before moving."
            ),
        },
        "high": {
            "message": (
                "Your feet are very unstable — significant movement on the holds "
                "suggests they are not properly engaged."
            ),
            "recommendation": (
                "Slow down and focus on your footwork. Smear or edge deliberately, "
                "and do not move until you feel your weight fully over the support foot."
            ),
        },
    },
    "locked_elbow_too_early": {
        "low": {
            "message": (
                "One arm may be reaching near-straight "
                "before your body has finished moving."
            ),
            "recommendation": (
                "Keep a slight bend in your elbow and let your legs "
                "do the heavy lifting to continue your upward progress."
            ),
        },
        "medium": {
            "message": (
                "Your arm is nearly straight too early, "
                "limiting your ability to continue the movement."
            ),
            "recommendation": (
                "Keep your elbow bent and use leg drive to move your body upward — "
                "save the extended arm position for when you are fully in place."
            ),
        },
        "high": {
            "message": (
                "Your arm is locking out very early, "
                "effectively stopping your upward movement."
            ),
            "recommendation": (
                "Focus on leg drive: push your hips up first, "
                "then extend your arm only when your body is already in position."
            ),
        },
    },
}

# Generic fallback for unknown codes
_FALLBACK: dict[str, dict[str, str]] = {
    "low": {
        "message": "A possible movement issue was detected.",
        "recommendation": "Review the annotated frame for context.",
    },
    "medium": {
        "message": "A movement issue was detected.",
        "recommendation": "Review the annotated frame for context.",
    },
    "high": {
        "message": "A significant movement issue was detected.",
        "recommendation": "Review the annotated frame for context.",
    },
}


def get_issue_texts(code: str, severity: str, confidence: float) -> tuple[str, str]:
    """Return (message, recommendation) for the given issue code and severity.

    If confidence is below the hedge threshold the message is prefixed with
    'Possible: ' to signal uncertain detection.
    """
    template = _TEMPLATES.get(code, _FALLBACK).get(severity, _FALLBACK["medium"])
    message = template["message"]
    recommendation = template["recommendation"]

    if confidence < _HEDGE_THRESHOLD:
        message = "Possible: " + message[0].lower() + message[1:]

    return message, recommendation
