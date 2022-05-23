# BentoML Documentation

This directory contains the sphinx source text for BentoML docs, visit
http://docs.bentoml.org/ to read the full documentation.

---

## Build Docs

If you haven't already, clone the BentoML Github repo to a local directory:

```bash
git clone https://github.com/bentoml/BentoML.git && cd BentoML
```

Install all dependencies required for building docs (mainly `sphinx` and its extension):

```bash
pip install -r requirements/docs-requirements.txt
```

Build the sphinx docs:

```bash
make clean html -C ./docs
```

The docs HTML files are now generated under `docs/build/html` directory, you can preview
it locally with the following command:

```bash
python -m http.server 8000 -d docs/build/html
```

And open your browser at http://0.0.0.0:8000/ to view the generated docs.


##### Watch Docs

On MacOS and Linux (UNIX-based system only), it is possible to watch documentation 
source changes and automatically rebuild the docs and refreshes the page in your 
browser.

Simply run the following command from BentoML project's root directory: 

```bash
make watch-docs
```

_Note: Accessing this feature requires `make` command_ 

> For Apple Silicon (M1), follow the latest suggested installation method for [PyEnchant](https://pyenchant.github.io/pyenchant/install.html). As of this writing there is no compatible arm64 version of pyenchant and the best way to install is the following commands:
>
> ```bash
> arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> arch -x86_64 /usr/local/bin/brew install enchant
> ```
> Make sure that PYENCHANT_LIBRARY_PATH is set to the location of libenchant. For MacOS make sure it has the dylib extension, otherwise the .so for Linux based systems.



## Writing Documentation


### Writing .rst (ReStructuredText) in BentoML docs

BentoML docs is built with Sphinx, which natively supports [ReStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

#### Adding Reference Links

When writing documentation, it is common to mention or link to other parts of the docs.

If you need to refer to a specific documentation page, use `:doc:` plus path to the 
target documentation file under the `docs/source/`. e.g.:

```rst
:doc:`tutorial`
:doc:`/frameworks/pytorch`
```

By default, this will show the title of the target document and link to it. You may also
change the title shown on current page:

```rst
:doc:`📖 Main Concepts <concepts/index>`
```

It is also possible to refer to a specific section of other document pages. We use the
[autosectionlabel sphinx plugin](https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html)
to generate labels for every section in the documentation.

For example:
```rst
:ref:`frameworks/pytorch:Section Title
```

#### Admonitions

A `note` section can be created with the following syntax:
```rst
.. note:: This is what the most basic admonitions look like.


.. note::
   It is *possible* to have multiple paragraphs in the same admonition.

   If you really want, you can even have lists, or code, or tables.
```

There are other admonition types such as `caution`, `danger`, `hint`, `important`, 
`seealso`, and `tip`. Learn more about it [here](https://pradyunsg.me/furo/reference/admonitions/).

#### Code Blocks

```rst
Code blocks in reStructuredText can be created in various ways::

    Indenting content by 4 spaces, after a line ends with "::".
    This will have default syntax highlighting (highlighting a few words and "strings").

.. code::

    You can also use the code directive, or an alias: code-block, sourcecode.
    This will have default syntax highlighting (highlighting a few words and "strings").

.. code:: python

    print("And with the directive syntax, you can have syntax highlighting.")

.. code:: none

    print("Or disable all syntax highlighting.")
```

There's a lot more forms of "blocks" in reStructuredText that can be used, as
seen in https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#literal-blocks.


#### Tabs

For most scenarios in BentoML docs, use the tabs view provided by `sphinx-design`:
https://sphinx-design.readthedocs.io/en/furo-theme/tabs.html

```rst
.. tab-set::

    .. tab-item:: Label1

        Content 1

    .. tab-item:: Label2

        Content 2
```

### Documenting Source Code

BentoML docs relies heavily on the Python docstrings defined together with the source
code. We ask our contributors to document every public facing APIs and CLIs, including
their signatures, options, and example usage. Sphinx can then use these inline docs to
generate API References pages.

Our `.rst` documents can also create reference to the inline docstrings. For example, a
`.rst` document can create a section made from a Python Class's docstring, using the
following syntax:

```rst
:class:`~bentoml.Service`
```

Similarly for functions and method:

```rst
:func:`~bentoml.models.list`
```
```rst
:meth:`~bentoml.Service.api`
```

BentoML codebase follows the [Google's docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
for writing inline docstring. Below are some examples.

#### Define arguments in a method

Arguments should be defined with ``Args:`` prefix, followed by a line with indentation. Each argument should be followed by
its type, a new indentation for description of given field. Each argument should follow the below definition:

```markdown
    Args:
        bento_name (:code:`str`):
            :class:`~bentoml.BentoService` identifier with name format :obj:`NAME:VERSION`.
            ``NAME`` can be accessed via :meth:`~bentoml.BentoService.name` and ``VERSION`` can
            be accessed via :meth:`~bentoml.BentoService.version`
```

For optional arguments, follow the following syntax. For example a function ```func()``` with following signature:

```python
def func(x: str=None, a: Optional[bool]=None):
    ...
```

then documentation should look like:

```markdown
    Args:
        x (:code:`str`, `optional`):
            Description of x ...
        a (`bool`, `optional`):
            Description of a ...
```

#### Define a multiline code block in a method

Make sure to define something like:
```markdown
Example::

    # example code here
    # ...
```

The ```Example``` can be replaced with any word of choice as long as there are two semicolons following. Read more about [``doctest``](https://docs.python.org/3/library/doctest.html)

#### Define a return block in a method

If a function returns value, returns should be defined with ``Returns:``, followed by a line with indentation. The first line
should be the type of the return, followed by a line return. An example for a return statement:

```markdown
    Returns:
        :obj:`Dict[str,str]` with keys are :class:`~bentoml.BentoService` nametag following with saved bundle path.
```
