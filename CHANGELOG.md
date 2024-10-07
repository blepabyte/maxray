# CHANGELOG

## v0.7.0 (2024-10-07)

### Documentation

* docs: update README ([`7dcc70f`](https://github.com/blepabyte/maxray/commit/7dcc70fa21e637bdf8e321de1db07fa7a8a40331))

### Feature

* feat: draw runtime callgraph ([`9eb6619`](https://github.com/blepabyte/maxray/commit/9eb6619c8c7cfda54fdebde36c21102817fb47c4))

* feat: capture and rerun inators ([`60bbb02`](https://github.com/blepabyte/maxray/commit/60bbb024108b5afc3285f0d3b75af21871f2fcc5))

* feat: custom template builder ([`802ddb3`](https://github.com/blepabyte/maxray/commit/802ddb3291ef5d3c5863296a00d0e76f2a061f08))

* feat: composable inators

- support construction from CLI arguments
- allow extending log capture mechanism
- remove `call_count` from FnContext ([`6e8991d`](https://github.com/blepabyte/maxray/commit/6e8991dd8bdfb2175df67b9590d43f88fb0698a6))

* feat: `entered` prop and nocompile context ([`9c8c59a`](https://github.com/blepabyte/maxray/commit/9c8c59a1a624fea0a8fadce51811f1775cb0e556))

### Fix

* fix: chained assignment unpacking ([`768b3df`](https://github.com/blepabyte/maxray/commit/768b3df05e87ed6630e761f9921411ec08a88562))

* fix: staticmethod transforms only ran on the first time ([`eae0fe9`](https://github.com/blepabyte/maxray/commit/eae0fe9156f003339c299680a96a010183634e49))

### Refactor

* refactor: NodeContext -&gt; RayContext ([`90ae0ee`](https://github.com/blepabyte/maxray/commit/90ae0ee5ab98f474c391bd662c4b27d2ca04bf9d))

* refactor: replace NodeContext with higher-level Ray interface ([`031fec4`](https://github.com/blepabyte/maxray/commit/031fec442f480893b3a9c2980635aecdad0ed625))

### Unknown

* iter: bind internal context for logging ([`8cc4c96`](https://github.com/blepabyte/maxray/commit/8cc4c96c183791ad94dda283068513cde7850e87))

## v0.6.0 (2024-09-11)

### Feature

* feat: more context props (call, iterate, return) ([`7b51b32`](https://github.com/blepabyte/maxray/commit/7b51b324ccaa03ca789e2a95908c257c9458b6cb))

### Fix

* fix: pass_local_scopes now globally applied ([`7e61361`](https://github.com/blepabyte/maxray/commit/7e61361e93dc3506200f2f64683e0eff8b161b59))

### Refactor

* refactor(runner): display and control flow logic in templates ([`8db73d9`](https://github.com/blepabyte/maxray/commit/8db73d9b669c7226c693235a8a792e66c3846aaf))

* refactor: separate out rich/display from core runner impl ([`cc795cf`](https://github.com/blepabyte/maxray/commit/cc795cf2c2ac1a3ec2c2beb6e6496edef2780c60))

## v0.5.0 (2024-09-03)

### Chore

* chore: comment out lines in template ([`a6603cb`](https://github.com/blepabyte/maxray/commit/a6603cb3bbe8a3fa56193cf62e0b20d8de687ed9))

### Feature

* feat: print traceback on exit ([`1b04de6`](https://github.com/blepabyte/maxray/commit/1b04de656bb45506ea8dd73ef6c43f64903579ed))

* feat: inator template for source overlays ([`2af8805`](https://github.com/blepabyte/maxray/commit/2af880557cb4ea652c1e34c524251d860bdd2bf6))

* feat: dashboard for interactive watch mode ([`20218d4`](https://github.com/blepabyte/maxray/commit/20218d488452ba89f7df2a2f2b25e77810f7f84c))

* feat: propagate more assignment info in `NodeContext.props` ([`869db4c`](https://github.com/blepabyte/maxray/commit/869db4c1d148a2eb2b6007eade79282b21d4d7ce))

### Fix

* fix: update scripts in pyproject.toml ([`51b6996`](https://github.com/blepabyte/maxray/commit/51b69963d8c4c7c4e28b45b39dfcb89c43da022e))

* fix: forbid metaclasses in transforms ([`c3cafaf`](https://github.com/blepabyte/maxray/commit/c3cafaf635770c1f620455d185da351cb5067460))

* fix: relative imports in modules ([`05ff21b`](https://github.com/blepabyte/maxray/commit/05ff21b6cc713f794806242d4ba21dafce4abf49))

* fix: flag exception state in xpy via return code ([`26d1390`](https://github.com/blepabyte/maxray/commit/26d1390d0275d98d0bcb8368b28f4c96aa444990))

* fix: support BinOp nodes ([`4491a38`](https://github.com/blepabyte/maxray/commit/4491a3869a89234195abf8d0fc91be7358d0e447))

* fix: unbound methods called with correct scope ([`3835f26`](https://github.com/blepabyte/maxray/commit/3835f26e8d2053d547820a1e84d33975a151d9af))

### Refactor

* refactor: logs_cli -&gt; runner module and saner control flow ([`cbddfd5`](https://github.com/blepabyte/maxray/commit/cbddfd5aa285573b42487ab7b7650ff8ead8a093))

* refactor: move runtime impl into RuntimeHelper ([`34694be`](https://github.com/blepabyte/maxray/commit/34694bea2638780a0b377b1779e70283b1e8629a))

## v0.4.0 (2024-08-15)

### Chore

* chore: add package name as script to work with pipx/uv ([`b70b3ee`](https://github.com/blepabyte/maxray/commit/b70b3ee1298e62586e4932594dfcb8f39b10ebf7))

### Feature

* feat(capture): stateful watching with -W ([`6b16aed`](https://github.com/blepabyte/maxray/commit/6b16aed27272ee1a7e88a41ac3dc61158485fa69))

* feat: improved method patching correctness ([`a5c991e`](https://github.com/blepabyte/maxray/commit/a5c991ed2e2a575acd480a7f931b42555e7d2a39))

### Fix

* fix: __init__ and __call__ are not mutually exclusive ([`89ba42c`](https://github.com/blepabyte/maxray/commit/89ba42cbfbeec31c41dfb6ebb55016a6a46c7686))

* fix: transform control flow and relative package import ([`aaee9f0`](https://github.com/blepabyte/maxray/commit/aaee9f07cf01e74b1cfeee3db913bb2e4a8ac299))

* fix: scope should reflect changes to module state ([`0ee134e`](https://github.com/blepabyte/maxray/commit/0ee134eb3abe759508d9c7ba09f54d2d38d63e49))

* fix: support `global` ([`38387be`](https://github.com/blepabyte/maxray/commit/38387be9ddcfed8740f58fbfaf4c2557cc0cd9f9))

### Refactor

* refactor: better data model for FunctionStore ([`7863b3a`](https://github.com/blepabyte/maxray/commit/7863b3a588896ad1b1587ae2c822ef5951053dcc))

### Test

* test: more transform cases ([`b528f67`](https://github.com/blepabyte/maxray/commit/b528f6763c2c4513b241a71875746842e9185a65))

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
