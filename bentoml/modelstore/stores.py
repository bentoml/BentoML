# file structure
#
# $BENTOML_HOME/model_store/{defined_name}/{version}/artifact.save() content
#
# path to parse for artifact.save: $BENTOML_HOME/model_store/{defined_name}/{versions}


class _LocalStores:
    """
    Manages versions of saved artifacts for local development workflow
    and online serving within BentoBundle
    """
