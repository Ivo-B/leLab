import { CameraLens } from "./cameraLens";

/**
 * Draws a `<video>` through the fisheye lens model on the GPU. The fragment shader is `lensSource` in
 * `cameraLens.ts` and `source_coordinates` in `lelab/lens.py`, written out in GLSL: keep the three in step.
 */
export interface LensRenderer {
  draw(video: HTMLVideoElement, lens: CameraLens): void;
  dispose(): void;
}

const VERTEX_SHADER = `
attribute vec2 corner;
varying vec2 position;
void main() {
  // Pixel space with y pointing down, like the camera picture.
  position = vec2((corner.x + 1.0) / 2.0, (1.0 - corner.y) / 2.0);
  gl_Position = vec4(corner, 0.0, 1.0);
}`;

export const FRAGMENT_SHADER = `
precision highp float;
uniform sampler2D picture;
uniform vec2 size;
uniform vec3 lens; // theta_max (radians), bend, zoom
varying vec2 position;
void main() {
  float halfDiagonal = 0.5 * length(size);
  vec2 offset = (position * size - 0.5 * size) / halfDiagonal;
  float rho = length(offset);
  float thetaMax = lens.x;
  float thetaDmax = thetaMax * (1.0 + lens.y * thetaMax * thetaMax);
  float theta = atan(rho * thetaDmax / lens.z);
  float rSource = theta * (1.0 + lens.y * theta * theta) / thetaDmax;
  float scale = rho > 1e-6 ? rSource / rho : 1.0 / lens.z;
  vec2 source = (0.5 * size + offset * scale * halfDiagonal) / size;
  if (source.x < 0.0 || source.x > 1.0 || source.y < 0.0 || source.y > 1.0) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    gl_FragColor = vec4(texture2D(picture, source).rgb, 1.0);
  }
}`;

function compile(gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.warn("Lens shader failed to compile", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

/** A renderer drawing into this canvas, or null when WebGL is not available. */
export function createLensRenderer(canvas: HTMLCanvasElement): LensRenderer | null {
  const gl = canvas.getContext("webgl", { premultipliedAlpha: false }) as WebGLRenderingContext | null;
  if (!gl) return null;
  const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  const program = gl.createProgram();
  if (!vertex || !fragment || !program) return null;
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.warn("Lens shader failed to link", gl.getProgramInfoLog(program));
    return null;
  }
  gl.useProgram(program);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const corner = gl.getAttribLocation(program, "corner");
  gl.enableVertexAttribArray(corner);
  gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  // Video frames are rarely a power of two: no mipmaps, no repeat.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  const size = gl.getUniformLocation(program, "size");
  const lensUniform = gl.getUniformLocation(program, "lens");

  return {
    draw(video, lens) {
      const { videoWidth: width, videoHeight: height } = video;
      if (!width || !height) return;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
      gl.uniform2f(size, width, height);
      gl.uniform3f(lensUniform, (lens.fov / 2) * (Math.PI / 180), lens.bend, lens.zoom);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
    dispose() {
      gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
    },
  };
}
