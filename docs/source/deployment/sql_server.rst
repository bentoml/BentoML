Deploying to SQL Server Machine Learning Services
=================================================

Microsoft's Machine Learning Services is a feature in SQL Server that gives the ability to run Python and R scripts with relational data.
It allows to use BentoML and other open-source packages, along with the Microsoft Python packages, for predictive analytics and machine learning. 
The scripts are executed in-database without moving data outside SQL Server or over the network.


This guide demonstrates how to serve a scikit-learn based iris classifier model with
SQL Server Machine Learning Services. The same deployment steps are also applicable for models
trained with other machine learning frameworks, see more BentoML examples :doc:`here <../examples>`.

=============
Prerequisites
=============

Before starting this guide, make sure you have the following:

* SQL server machine learning services installed: https://www.microsoft.com/en-in/sql-server/sql-server-downloads

* SSMS installed and connected to server: https://docs.microsoft.com/en-us/sql/ssms/download-sql-server-management-studio-ssms

* Python 3.7 or above and required PyPi packages: `bentoml` and `scikit-learn`

  * .. code-block: bash

          pip install bentoml scikit-learn

* `bentoml` installed in SQL server. In terminal change directory to 
  * .. code-block: bash
  
          $ SQL Server/PYTHON_SERVICES/SCRIPTS

          $ pip.exe install bentoml

As long as the server is not connected to remote compute, no server costs will be accumulated.



SQL Server deployment with BentoML
----------------------------------

Run the example project from the :doc:`quick start guide <../quickstart>` to create the
BentoML saved bundle for deployment:


.. code-block:: bash

    git clone git@github.com:bentoml/BentoML.git
    pip install -r ./bentoml/guides/quick-start/requirements.txt
    python ./bentoml/guides/quick-start/main.py

Verify the saved bundle created and get the latest path:

.. code-block:: bash

    $ bentoml get --print-location IrisClassifier:latest 


Start a new query and write the following command. This will enable running external scripts.

.. code-block:: bash

    sp_configure

    EXEC sp_configure 'external scripts enabled', 1

    RECONFIGURE WITH OVERRIDE

Create a new database to store the dataset for prediction.

.. code-block:: bash

    CREATE DATABASE irissql

    GO

Create new table in the database just created and put columns similar to the iris dataset. Data will be saved here later.

.. code-block:: bash

    USE irissql

    GO

    DROP TABLE IF EXISTS iris_data;

    GO

    CREATE TABLE iris_data (

    id INT NOT NULL IDENTITY PRIMARY KEY

    , "Sepal.Length" FLOAT NOT NULL, "Sepal.Width" FLOAT NOT NULL

    , "Petal.Length" FLOAT NOT NULL, "Petal.Width" FLOAT NOT NULL

    , "Species" VARCHAR(100) NOT NULL, "SpeciesId" INT NOT NULL

    );

Next create a procedure which works similar to a method(function) in python. List a set of steps that will get executed while using this procedure. Procedure `get_iris_dataset` will be used to insert values in the table.

.. code-block:: bash

    CREATE PROCEDURE get_iris_dataset

    AS

    BEGIN

    EXEC sp_execute_external_script @language = N'Python',

    @script = N'

    from sklearn import datasets

    iris = datasets.load_iris()

    iris_data = pandas.DataFrame(iris.data)

    iris_data["Species"] = pandas.Categorical.from_codes(iris.target, iris.target_names)

    iris_data["SpeciesId"] = iris.target

    ',

    @input_data_1 = N'',

    @output_data_1_name = N'iris_data'

    WITH RESULT SETS (("Sepal.Length" float not null, "Sepal.Width" float not null, "Petal.Length" float not null, "Petal.Width" float not null,

    "Species" varchar(100) not null, "SpeciesId" int not null));

    END;

    GO

Finally insert data into the table iris_data and execute the procedure get_iris_dataset.

.. code-block:: bash

    INSERT INTO iris_data ("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species", "SpeciesId")

    EXEC dbo.get_iris_dataset;



The last step is to create a procedure for model deployment and prediction. Create a procedure predict_species and as an external script run the bento saved bundle.

.. code-block:: bash

    import bentoml as usual and set the saved_path to the location where the bento bundle is saved.

    Load the bundle using bentoml.load(). Now use this model loaded from the saved bundle to make predictions and deploy the model. List all the input and output features.

    Here is the complete script

    CREATE PROCEDURE predict_species (@model VARCHAR(100))

    AS

    BEGIN

    `DECLARE @svm_model VARBINARY(max)`

    `EXECUTE sp_execute_external_script @language = N'Python'`

        `, @script = N'`
    import bentoml

    saved_path=r"C:\Program Files\Microsoft SQL Server\MSSQL15.NEWSERVER\bento_bundle"

    irismodel = bentoml.load(saved_path)

    species_pred = irismodel.predict(iris_data[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]])

    iris_data["PredictedSpecies"] = species_pred

    OutputDataSet = iris_data[["id","SpeciesId","PredictedSpecies"]]

    print(OutputDataSet)

    ' , @input_data_1 = N'select id, "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "SpeciesId" from iris_data'

        `, @input_data_1_name = N'iris_data'`
        
        `, @params = N'@svm_model varbinary(max)'`
        
        `, @nb_model = @svm_model`

    `WITH RESULT SETS((`
    
                `"id" INT`
            
            `, "SpeciesId" INT`

            `, "SpeciesId.Predicted" INT`
            
            ` ));`
    END;

    GO

The procedure is ready now. Deploy it using Execute predict_species 'SVM';

.. code-block:: bash

    EXECUTE predict_species 'SVM';

    GO


After executing the final query you can see the predictions in form of a table. 

.. code-block:: bash

    SELECT * FROM predict_species;

To disconnect from the server, click the disconnect icon on the left panel under Object Explorer in SSMS.
The model is served with SQL server easily with the help of BentoML.

.. spelling::

    analytics
    exe
