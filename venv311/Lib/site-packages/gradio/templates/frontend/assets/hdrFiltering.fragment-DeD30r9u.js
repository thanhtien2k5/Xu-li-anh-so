import{S as i}from"./index-Bmc01_3m.js";import"./helperFunctions-0NFFyukF.js";import"./hdrFilteringFunctions-Dw2FVAR5.js";import"./pbrBRDFFunctions-CGRU0824.js";import"./index-yXVd8EKt.js";const r="hdrFilteringPixelShader",e=`#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform float alphaG;uniform samplerCube inputTexture;uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=radiance(alphaG,inputTexture,direction,vFilteringInfo);gl_FragColor=vec4(color*hdrScale,1.0);}`;i.ShadersStore[r]||(i.ShadersStore[r]=e);const c={name:r,shader:e};export{c as hdrFilteringPixelShader};
