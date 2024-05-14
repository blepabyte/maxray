# CHANGELOG



## v0.2.0 (2024-05-14)

### Documentation

* docs: fix project urls ([`e842dfe`](https://github.com/blepabyte/maxray/commit/e842dfe159cc3210c30de43f340fc5bc1ed4a276))

### Feature

* feat: track return nodes, optionally collecting the local scope on exit ([`8afe9ad`](https://github.com/blepabyte/maxray/commit/8afe9adcf9ddd186d9978cd35ba95ab05dbd3b26))

* feat: add counter for n-th function invocation to node context ([`0d51259`](https://github.com/blepabyte/maxray/commit/0d512592858ec20b7c8b05aba63b31413de2f9ce))

### Fix

* fix: remove kwonly and return type annotations ([`f0f66d0`](https://github.com/blepabyte/maxray/commit/f0f66d046f3558aeae6ad6c2510839ad670e3db5))


## v0.1.1 (2024-05-04)

### Fix

* fix: ctypes module check ([`883ee4f`](https://github.com/blepabyte/maxray/commit/883ee4fb93fefa5429cd22ff66ea4de36caa3227))


## v0.1.0 (2024-04-21)

### Chore

* chore: add project metadata ([`7777d01`](https://github.com/blepabyte/maxray/commit/7777d01f1b58ac86501daad0db392c8204181e5b))

### Ci

* ci: fix actions yaml syntax ([`1a83801`](https://github.com/blepabyte/maxray/commit/1a8380120a78988c262b8098d6c5300e21dd99b0))

* ci: add semantic release ([`faec1c7`](https://github.com/blepabyte/maxray/commit/faec1c7ff25b6bdaebbc0630a6d0d2031161177f))

* ci: add test workflow ([`b230176`](https://github.com/blepabyte/maxray/commit/b230176d998643c20d0e5961e12be5d24425b505))

### Documentation

* docs: update README ([`cd6ce55`](https://github.com/blepabyte/maxray/commit/cd6ce55e3d74e92187c3c584f1a1c71d6a0f070f))

* docs: add README ([`3894fbe`](https://github.com/blepabyte/maxray/commit/3894fbe2f9ba39ceaf6ae85e93336ab833b7fa53))

### Feature

* feat: handle async functions

Also fixes for `match` handling, functools wrapping, and multiple decorators ([`71029d0`](https://github.com/blepabyte/maxray/commit/71029d058e1a6fd39f51439c63968ab8563ba56e))

* feat: implement transform, xray, maxray ([`6427aef`](https://github.com/blepabyte/maxray/commit/6427aef210e118944c83dc89556f9a8ba9b72650))

### Fix

* fix: keep exact original source code locations ([`2d48e9a`](https://github.com/blepabyte/maxray/commit/2d48e9ad6786f3b6459e9acbfb5741812d835589))

### Refactor

* refactor: function descend logic ([`26bd6fe`](https://github.com/blepabyte/maxray/commit/26bd6fed9739c4ca3ca2ddb39324429f0b99bbd4))

### Test

* test: decorators and callable hashing ([`db4fd51`](https://github.com/blepabyte/maxray/commit/db4fd51964decf84d786a3efe6ae7a66955a07b8))

* test: check side effects ([`9facb09`](https://github.com/blepabyte/maxray/commit/9facb09a09f9cb57a50ef329ff362076127d012e))
