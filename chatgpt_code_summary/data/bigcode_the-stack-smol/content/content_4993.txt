DEPS = [
  'archive',
  'chromium',
  'chromium_android',
  'depot_tools/bot_update',
  'depot_tools/gclient',
  'depot_tools/tryserver',
  'file',
  'recipe_engine/path',
  'recipe_engine/platform',
  'recipe_engine/properties',
  'recipe_engine/python',
  'recipe_engine/step',
  'trigger',
]


# TODO(phajdan.jr): provide coverage (http://crbug.com/693058).
DISABLE_STRICT_COVERAGE = True
