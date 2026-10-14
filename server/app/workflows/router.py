from app.schemas.ticket import TicketClassification


def route_ticket(ticket: TicketClassification) -> dict:

    # 1. Handle requests the classifier cannot understand
    if not ticket.query_understood:
        return {
            "next_route": "clarify_intent",
            "message": "Please explain your issue in more detail."
        }

    # 2. Handle requests requiring human review
    if ticket.requires_human_review:
        return {
            "next_route": "human_review",
            "message": "This issue requires human review."
        }

    # 3. Ask for required missing information
    if ticket.missing_information:
        return {
            "next_route": "request_information",
            "message": ticket.clarification_question,
            "missing_information": ticket.missing_information
        }

    # 4. Route to the next approved workflow
    if ticket.action_ready:
        return {
            "next_route": ticket.next_route,
            "recommended_action": ticket.recommended_action
        }

    # 5. Safe fallback
    return {
        "next_route": "manual_review",
        "message": "The ticket needs further assessment."
    }