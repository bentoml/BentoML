# BentoML Documentation

This directory contains the sphinx source text for BentoML docs, visit
http://docs.bentoml.org/ to read the full documentation.

---

**NOTE**:
All of the below `make` commands should be used under `bentoml` root directory.

To generate the documentation, make sure to install all dependencies (mainly `sphinx` and its extension):

```bash
» make install-docs-deps
```

Once you have `sphinx` installed, you can build the documentation and enable watch on changes:
```bash
» make watch-docs
```

## Documentation specification

`bentoml/BentoML` follows [Google's docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
mostly written in [ReStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)

### Writing source documentation

Value should either put around ``` ``double backticks`` ```, or put into a ``` :code:`codeblock` ``` or using the object syntax ``` :obj:`class` ```.
When mentioning a `class` it is recommended to use the ``` :class:`syntax` ``` as mentioned class will be linked by Sphinx:
  ```markdown
  :class:`~bentoml.BentoService`
  ```
When mentioning a `function`, it is recommended to use the ``` :func:`syntax` ``` as mentioned function will be linked by Sphinx:
```markdown
:func:`~bentoml.yatai.client.func`
```
When mentioning a method, it is recommended to use the ``` :meth:`syntax` ``` as mentioned method will be linked by Sphinx:
```markdown
:meth:`~bentoml.BentoService.method`
```
  
#### Define arguments in a method

Arguments should be defined with ``Args:`` prefix, followed by a line with indentation. Each argument should be followed by
its type, a new indentation for description of given field. Each argument should follow the below definition:

```markdown
    Args:
        bento_name (`str`):
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
        x (`str`, `optional`):
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

#### Tips and Tricks

Header level hierarchy in rst:

```text
1 -
2 ~
3 ^
4 =
5 "
```