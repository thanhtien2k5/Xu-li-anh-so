import{S as e}from"./index-Bmc01_3m.js";const d="pointCloudVertex",i=`#if defined(POINTSIZE) && !defined(WEBGPU)
gl_PointSize=pointSize;
#endif
`;e.IncludesShadersStore[d]||(e.IncludesShadersStore[d]=i);
