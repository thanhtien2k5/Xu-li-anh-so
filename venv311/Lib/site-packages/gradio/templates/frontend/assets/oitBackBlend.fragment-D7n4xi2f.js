import{S as r}from"./index-Bmc01_3m.js";import"./index-yXVd8EKt.js";const o="oitBackBlendPixelShader",e=`precision highp float;uniform sampler2D uBackColor;void main() {glFragColor=texelFetch(uBackColor,ivec2(gl_FragCoord.xy),0);if (glFragColor.a==0.0) { 
discard;}}`;r.ShadersStore[o]||(r.ShadersStore[o]=e);const l={name:o,shader:e};export{l as oitBackBlendPixelShader};
