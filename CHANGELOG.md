# CHANGELOG

## v0.3.0 (2024-06-25)

### Chore

* chore: add `rich` dependency ([`a0a3489`](https://github.com/blepabyte/maxray/commit/a0a34891b28cdf5e3f40feabd00d02aca8fd12de))

* chore: add pyarrow dependency ([`2b465ca`](https://github.com/blepabyte/maxray/commit/2b465ca105cacd638bf15092abef909edda445ac))

* chore: rename CLI entrypoint from `capture-logs` to `xpy` ([`6efda68`](https://github.com/blepabyte/maxray/commit/6efda68f95f122e565b1d2a80708cb801c3cfbfb))

### Ci

* ci: give up on OIDC and go back to using API token ([`038d4e7`](https://github.com/blepabyte/maxray/commit/038d4e7da427d50b2301034c21cebf18a76fdfe8))

* ci: run tests for PRs ([`0b4846a`](https://github.com/blepabyte/maxray/commit/0b4846a4c4f799ce04414ed691737fcb8e794c35))

### Documentation

* docs: really fix project urls this time ([`982af59`](https://github.com/blepabyte/maxray/commit/982af590525a08f148f6d3f7f03e32cab555aa5f))

### Feature

* feat: pass EndOfScript marker to inators in ScriptRunner ([`119d4a0`](https://github.com/blepabyte/maxray/commit/119d4a006ca84cfc617139170fdba0f061784d02))

* feat: ScriptRunner interface w/ CLI support ([`7800351`](https://github.com/blepabyte/maxray/commit/780035111e50cd99923c64c34b938c0568a425ba))

* feat: hot-reloading of transform funcs ([`d376135`](https://github.com/blepabyte/maxray/commit/d376135076aa3ad70079ac03bf8a0513ef616020))

* feat: support `-m` module-level scripts ([`af8680d`](https://github.com/blepabyte/maxray/commit/af8680d1819448a7a7574b2ff86e5bcde6051837))

* feat: convenience command to capture logs from external scripts ([`3006911`](https://github.com/blepabyte/maxray/commit/30069112d762b91d103cd1a049df4c9cc5e0e0b4))

* feat(capture): export logs to Arrow ([`ba5e0c5`](https://github.com/blepabyte/maxray/commit/ba5e0c5aac44bfaf80efdfc05a8b32b2378c617f))

* feat: assign transformed functions a UUID ([`9557d68`](https://github.com/blepabyte/maxray/commit/9557d689c3002b3f2d3deeea480516cfed0e6f19))

* feat: patch __init__ and __call__ ([`de4e1f0`](https://github.com/blepabyte/maxray/commit/de4e1f06f76872be5188623abc1e6fe414613432))

### Fix

* fix: re-ordering, and guard against exceptions ([`c9bc661`](https://github.com/blepabyte/maxray/commit/c9bc661b66de5cdb3b0979fedd223f3c0523f55b))

* fix: memory leak ([`476ba7b`](https://github.com/blepabyte/maxray/commit/476ba7bb57a6883e28a63e28999c5ba7b388098e))

* fix(transform): order scope layers to prevent namespace pollution

Fixes bug where np.load errors with &#34;numpy.AxisError: axis 6 is out of bounds for array of dimension 0&#34; because `min` was dispatching to `np.min` due to prior `@set_module` decorator. ([`149088e`](https://github.com/blepabyte/maxray/commit/149088ebce84e350eff7d266ef82c27b4befea96))

* fix: re-order writer lock to support patching ([`87ce1ba`](https://github.com/blepabyte/maxray/commit/87ce1ba91238ed83b1ca2a09839695b61b9bba38))

* fix: descend logic and lru_cache support ([`4be7e71`](https://github.com/blepabyte/maxray/commit/4be7e7183657218d0993b2fddf21e8798cca05ec))

* fix: super() and method descriptor patching ([`cbd7b48`](https://github.com/blepabyte/maxray/commit/cbd7b485612ea34081af8d194dd6a395176db606))

### Refactor

* refactor: function store for serialising source code and metadata ([`c3d3cec`](https://github.com/blepabyte/maxray/commit/c3d3cec47e23481818dcdb955293b3b20390c384))

### Test

* test: remove implicit use of pandas ([`0b01e6a`](https://github.com/blepabyte/maxray/commit/0b01e6a772a37f6c9581dd6de6071b32dff348e6))

* test: script interface ([`1e65c87`](https://github.com/blepabyte/maxray/commit/1e65c87d0054796819f66e39caf46d783b2a1de3))

* test: add Python 3.13 to test matrix ([`fb0041f`](https://github.com/blepabyte/maxray/commit/fb0041fa59e6e759870405847464d119caaec125))

* test: add missing package checks ([`4925443`](https://github.com/blepabyte/maxray/commit/4925443b7cd569fa61121d7e1f6d23d597ffcc49))

* test: external package compat for numpy, pandas, torch ([`2e93ce4`](https://github.com/blepabyte/maxray/commit/2e93ce4213f4d7925e5c00c069efda7135b8c78e))

### Unknown

* Merge pull request #5 from blepabyte/06-17-feat_hot-reloading_of_transform_funcs

feat: CLI and programmatic interface to script runner ([`56c420b`](https://github.com/blepabyte/maxray/commit/56c420b1a940b103bb3137238512e535b06b5c37))

* Revert &#34;test: add Python 3.13 to test matrix&#34;

poetry/pyarrow/etc. build issues ([`21402e4`](https://github.com/blepabyte/maxray/commit/21402e49bb2d5c9ccea4bc87b77426f429439006))

* Merge pull request #4 from blepabyte/06-10-feat_capture_export_logs_to_arrow

feat(capture): export logs to Arrow ([`7e6b9d8`](https://github.com/blepabyte/maxray/commit/7e6b9d817ef8c427040ef8bf8ae0b8a52b91aef1))

* Merge pull request #3 from blepabyte/06-17-fix_transform_order_scope_layers_to_prevent_namespace_pollution

fix(transform): order scope layers to prevent namespace pollution ([`377e1ac`](https://github.com/blepabyte/maxray/commit/377e1ac7ab7c88af865cf231ae1df1a903ee4eab))

* Merge pull request #2 from blepabyte/06-10-feat_assign_transformed_functions_a_uuid

feat: assign transformed functions a UUID ([`4d60721`](https://github.com/blepabyte/maxray/commit/4d6072122dbfac0f659b63d4f5f8952291e30e9e))

* Merge pull request #1 from blepabyte/06-10-fix_descend_logic_and_lru_cache_support

fix: descend logic and lru_cache support ([`9d0fbde`](https://github.com/blepabyte/maxray/commit/9d0fbdebcba60e2b7eb86deff536f413f3195213))

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
