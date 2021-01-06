from bentoml.utils import cached_contextmanager, cached_property


def test_cached_property():
    class A:
        def __init__(self):
            self.counter = 0

        @cached_property
        def a(self):
            self.counter += 1
            return 'a'

    a = A()
    assert a.a == a.a
    assert a.counter == 1

    class B(A):
        @property
        def a(self):  # pylint: disable=invalid-overridden-method
            _a = super(B, self).a
            return _a

    b = B()
    assert b.a == b.a
    assert b.counter == 1

    class C(A):
        pass

    c = C()
    assert c.a == c.a
    assert c.counter == 1


def test_cached_contextmanager():
    class DockerRepo:
        def __init__(self, path):
            self.path = path

    class DockerImage:
        built_times = 0

        def __init__(self, repo):
            self.repo = repo
            self.id = self.built_times + 1
            self.__class__.built_times += 1
            self.deleted = False

        def delete(self):
            self.deleted = True

        def __del__(self):
            assert self.deleted

    def build_docker_image(repo):
        return DockerImage(repo)

    repo = DockerRepo("1")

    image = build_docker_image(repo)
    image.delete()
    assert DockerImage.built_times == 1

    image = build_docker_image(repo)
    image.delete()
    assert DockerImage.built_times == 2

    @cached_contextmanager("{repo.path}")
    def reusable_docker_image(repo):
        image = build_docker_image(repo)
        yield image
        image.delete()

    with reusable_docker_image(repo) as image1:
        with reusable_docker_image(repo) as image2:
            assert image1.id == image2.id
    assert DockerImage.built_times == 3

    repo2 = DockerRepo("2")
    with reusable_docker_image(repo) as image1:
        with reusable_docker_image(repo2) as image2:
            assert image1.id != image2.id
    assert DockerImage.built_times == 5
