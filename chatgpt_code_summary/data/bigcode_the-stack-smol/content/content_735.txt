from sepal_ui import sepalwidgets as sw
from ipywidgets import dlink

from component import parameter as cp


class ParamTile(sw.Card):
    def __init__(self, model):

        # read the model
        self.model = model

        # add the base widgets
        self.close = sw.Icon(children=["mdi-close"], small=True)
        self.title = sw.CardTitle(
            class_="pa-0 ma-0", children=[sw.Spacer(), self.close]
        )

        # create the widgets
        self.w_target = sw.Select(
            small=True,
            items=[{"text": f"{i+1}0%", "value": i + 1} for i in range(cp.nb_target)],
            v_model=model.target,
            label="target",
            dense=True,
        )
        self.w_weight = sw.Select(
            small=True,
            items=[i + 1 for i in range(cp.nb_weight)],
            v_model=model.weight,
            label="weight",
            dense=True,
        )

        # link the widgets to the model
        self.model.bind(self.w_target, "target").bind(self.w_weight, "weight")

        # create the object
        super().__init__(
            max_width="500px",
            class_="pa-1",
            children=[self.title, self.w_target, self.w_weight],
            viz=False,
            disabled=False,
        )

        # add javascript events
        self.close.on_event("click", lambda *args: self.hide())
        dlink((self, "disabled"), (self, "loading"))

    def reset(self):

        self.w_target.v_model = None
        self.w_weight.v_model = None

        self.hide()

        return
