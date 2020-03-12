Core Concepts
=============

The main idea behind BentoML is that Data Science team should be shipping prediction
services instead of shipping pickled models. BentoService is the base component for
building prediction services using BentoML.


Each BentoService can contain multiple models via the BentoML :code:`Artifact` class,
and can define multiple APIs for accessing this service. Each API should specify a type
of :code:`Handler`, which defines the expected input data format for this API, most
commonly we see the use of :code:`DataframeHandler`, :code:`TensorHandler`, and
:code:`JsonHandler`.


Once you've trained an ML model, you can use the :code:`pack` method to bundle it with a
BentoService instance, and save the BentoService to a file directory. In the process
of :code:`BentoService#save`, BentoML serializes the model based on the ML
training framework you're using, automatically extracts all the pip dependencies
required by your BentoService class, and saves all the code, serialized model files,
and requirements.txt etc into a file directory, which we call it a SavedBundle.


BentoML also provide a mechanism for easy model management. It keeping track of all the
BentoService SavedBundle you've created and provide web UI and CLI for management. By
default BentoML keeps track of all the model files and metadata in your local file
system. But it is also possible to run a BentoML server that stores those data in the
cloud, and allow your ML team to easily share, find and use each others' models.


Below you can find more detailed introductions to the concepts mentioned above and how
to use and customize them for your ML model.


Introducing BentoService
------------------------

BentoService is the base class for creating prediction service with BentoML.




Packaging Model Artifacts
-------------------------


Using API Handlers
------------------


Using SavedBundle
-----------------


Model Management
----------------


Deploying BentoService
----------------------



