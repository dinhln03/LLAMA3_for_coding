label = 'spss'


def add_steps(steps: list, pipeline_id: str, config: dict) -> list:
    steps.append(('spss.add_spss', {
        'source': config['url']
    }))

    return steps
