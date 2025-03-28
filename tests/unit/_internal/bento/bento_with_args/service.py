import bentoml

args = bentoml.use_arguments()


@bentoml.service(labels={"foo": args.label})
class MyService:
    pass
