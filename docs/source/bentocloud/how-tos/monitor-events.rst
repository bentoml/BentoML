==============
Monitor events
==============

In BentoCloud, events provide a historical view of activities that have occurred within your resources.
You can use these events to track changes, monitor the state of your resources, and troubleshoot when issues arise.
Events can be related to different resources such as Models, Bentos, Deployments, and Clusters.

Types of events
===============

1. **Model:** These events involve activities related to the models you've stored in BentoCloud.
   It could be creating a new model, updating a model, or deleting a model.
2. **Bento:** Bento events focus on actions taken on Bentos, like creating, updating, or deleting a Bento.
3. **Deployment:** Events under this category involve deployment activities such as deploying a new Bento,
   updating a deployment, or deleting a deployment.
4. **Cluster:** Cluster events represent changes in the configuration or state of your BentoCloud cluster.

Monitor events
==============

Via user interface
------------------

The simplest way to monitor these events is through the BentoCloud UI.
The platform provides an interface where you can easily track the events related to your resources in the `Events <http://cloud.bentoml.com/events>`_ Page.

.. image:: ../../_static/img/bentocloud/events-homepage.png
    :alt: BentoCloud Events Homepage

Via API
-------

For more advanced monitoring or to integrate event tracking with your own systems,
you can use the BentoCloud API. This allows you to programmatically fetch events and process them according to your needs.

Search and filter events
========================

BentoCloud provides powerful tools for searching and filtering events.
You can filter events by date, resource type, operation name, and creator.

For instance, you might want to see all events related to a particular model within a certain time range.
Or you might want to track all the actions performed by a certain user.

In addition to filtering, events are also searchable.
The search syntax is similar to GitHub's, allowing for advanced search queries.
For example, to see all events created by a specific user, you could use ``creator:username``.
For more information on how to use the search syntax,
refer to `GitHub's search syntax documentation <https://docs.github.com/en/search-github/searching-on-github/searching-issues-and-pull-requests>`_.
