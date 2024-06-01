import requests
import json
from .config import auth_token, base_url
from .recommendation_client import RecommendationClient
from .json_parser import json_parser


class ExperimentClient(object):
    """Experiment Client Class

    This object defines a Thor experiment within the Python environment. In
    particular, an experiment is defined by its name, the date at which it was
    created, and the dimensions of the machine learning model. Moreover, an
    authentication token is required for requesting new parameter
    configurations, for submitting observations of parameters, for viewing
    pending parameter configurations and for obtaining the best configuration
    of parameters that has been evaluated so far.

    Parameters:
        identifier (int): A unique identifier that indicates which experiment
            on the server-side is being interacted with by the client.
        name (str): A name for the machine learning experiment. Consumers of the
            Thor service must have unique experiment names, so make sure all of
            your experiments are named different things!
        date (datetime): The datetime at which the experiment was created on the
            server side.
        dims (list of dictionaries): A list of dictionaries describing the
            parameter space of the optimization problem. Each dimension is given
            a name, a maximum value, a minimum value, and a dimension type that
            roughly describes how points are spaced.
        auth_token (str): String containing a user's specific API key provided
            by the Thor server. This is used to authenticate with the Thor
            server as a handshake that these experiments belong to a user and
            can be viewed and edited by them.
        base_url (str): String indicating the URL template for API calls.
    """
    def __init__(self, identifier, name, date, dims, auth_token=auth_token,
                 base_url=base_url):
        """Initialize parameters of the experiment client object."""
        self.experiment_id = identifier
        self.name = name
        self.date = date
        self.dims = dims
        self.auth_token = auth_token
        self.base_url = base_url

    def submit_observation(self, config, target):
        """Upload a pairing of a configuration alongside an observed target
        variable.

        Parameters:
            config (dictionary): A dictionary mapping dimension names to values
                indicating the configuration of parameters.
            target (float): A number indicating the performance of this
                configuration of model parameters.

        Examples:
            This utility is helpful in the event that a machine learning
            practitioner already has a few existing evaluations of the system at
            given inputs. For instance, the consumer may have already performed
            a grid search to obtain parameter values.

            Suppose that a particular experiment has two dimensions named "x"
            and "y". Then to upload a configuration to the Thor server, we
            proceed as follows:

            >>> d = {"x": 1.5, "y": 3.1}
            >>> v = f(d["x"], d["y"])
            >>> exp.submit_observation(d, v)
        """
        post_data = {
            "auth_token": self.auth_token,
            "experiment_id": self.experiment_id,
            "configuration": json.dumps(config),
            "target": target
        }
        result = requests.post(
            url=self.base_url.format("submit_observation"),
            json=post_data
        )
        return json_parser(result, self.auth_token)

    def create_recommendation(
            self,
            rand_prob=0.,
            n_models=5,
            description="",
            acq_func="expected_improvement",
            integrate_acq=True
    ):
        """Get a recommendation for a point to evaluate next.

        The create recommendation utility represents the core of the Thor
        Bayesian optimization software. This function will contact the Thor
        server and request a new configuration of machine learning parameters
        that serve the object of maximizing the metric of interest.

        Parameters:
            rand_prob (optional, float): This parameter represents that a random
                point in the input space is chosen instead of selecting a
                configuration of parameters using Bayesian optimization. As
                such, this parameter can be used to benchmark against random
                search and otherwise to perform pure exploration of the
                parameter space.
            n_models (optional, int): The number of Gaussian process models to
                sample using elliptical slice sampling. Setting this to a large
                number will produce a better characterization of uncertainty in
                the acquisition function.
            description (optional, str): An optional per-observation
                descriptor, potentially useful for identifying one observation
                among many others in a large experiment.  Defaults to "".
            acq_func (optional, str): A string specifying which acquisition
                function should be used to construct the newest recommendation.
                It can be useful to sometimes vary the acquisition function to
                enable exploitation towards the end of an experiment.
            integrate_acq (optional, bool): An indicator for whether or not we
                should construct an integrated acquisition function using models
                sampled from the posterior. The alternative is to not integrate
                and to return a single recommendation for each of the sampled
                models, of which there are `n_models`.

        Returns:
            RecommendationClient: A recommendation client object
                corresponding to the recommended set of parameters. If the
                acquisition function is not integrated, a list of
                RecommendationClient objects may be returned instead, one for
                each sampled model.
        """
        post_data = {
            "auth_token": self.auth_token,
            "experiment_id": self.experiment_id,
            "n_models": n_models,
            "rand_prob": rand_prob,
            "description": description,
            "acq_func": acq_func,
            "integrate_acq": integrate_acq
        }
        result = requests.post(
            url=self.base_url.format("create_recommendation"),
            json=post_data
        )
        recs = json_parser(result, self.auth_token, RecommendationClient)
        return recs[0] if len(recs) == 1 else recs

    def best_configuration(self):
        """Get the configuration of parameters that produced the best value of
        the objective function.

        Returns:
            dictionary: A dictionary containing a detailed view of the
                configuration of model parameters that produced the maximal
                value of the metric. This includes the date the observation was
                created, the value of the metric, and the configuration itself.
        """
        post_data = {
            "auth_token": self.auth_token,
            "experiment_id": self.experiment_id
        }
        result = requests.post(
            url=self.base_url.format("best_configuration"),
            json=post_data
        )
        return json_parser(result, self.auth_token)

    def pending_recommendations(self):
        """Query for pending recommendations that have yet to be evaluated.

        Sometimes client-side computations may fail for a given input
        configuration of model parameters, leaving the recommendation in a kind
        of "limbo" state in which is not being evaluated but still exists. In
        this case, it can be advantageous for the client to query for such
        pending observations and to evaluate them. This function returns a list
        of pending recommendations which can then be evaluated by the client.

        Returns:
            list of RecommendationClient: A list of
                recommendation client objects, where each element in the list
                corresponds to a pending observation.
        """
        post_data = {
            "auth_token": self.auth_token,
            "experiment_id": self.experiment_id
        }
        result = requests.post(
            url=self.base_url.format("pending_recommendations"),
            json=post_data
        )
        return json_parser(result, self.auth_token, RecommendationClient)

    @classmethod
    def from_dict(cls, dictionary, auth_token):
        """Create an experiment object from a dictionary representation. Pass
        the authentication token as an additional parameter.

        TODO:
            Can the authentication token be a return parameter?
        """
        return cls(
            identifier=dictionary["id"],
            name=dictionary["name"],
            date=dictionary["date"],
            dims=dictionary["dimensions"],
            auth_token=auth_token
        )
