# Python  extension planning rules

The Python layer is future work and should not drive the first implementation.

Current priority:
1. correct C++ reference behavior;
2. correct CUDA behavior against the C++ reference;
3. numerical test coverage;
4. stable C++ API;
5. Python binding layer.

When discussing the Python layer, Claude should focus on:
- API shape;
- array ownership;
- dtype and device semantics;
- zero-copy possibilities;
- error handling;
- packaging;
- testing against the C++/CUDA reference.

Claude should not write binding code until the C++/CUDA version is tested thoroughly.
