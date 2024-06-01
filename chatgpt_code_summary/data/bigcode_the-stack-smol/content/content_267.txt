# -*- coding: utf8 -*-


def filter_event(event, happening_before):
    """Check if the following keys are present. These
    keys only show up when using the API. If fetching
    from the iCal, JSON, or RSS feeds it will just compare
    the dates
    """
    status = True
    visibility = True
    actions = True
    if 'status' in event:
        status = event['status'] == 'upcoming'
    if 'visibility' in event:
        visibility = event['visibility'] == 'public'
    if 'self' in event:
        actions = 'announce' not in event['self']['actions']

    return (status and visibility and actions and
        event['time'] < happening_before)

