import { expect, test } from "vitest";
import { previewUrl, snapshotUrl } from "./cameraApi";
import { CameraConfig, cameraConfigurationError, cameraRotation, normalizeCameraConfig, rotatedDimensions, rotationForCamera, serializeCameras } from "./cameraConfig";
const camera: CameraConfig = { id: "wrist", name: "wrist", camera_id: "0x21100000c456367", type: "opencv", width: 640, height: 480, fps: 30 };

test("turning colour correction off is sent with the session and an omitted flag stays on", () => {
  expect(serializeCameras([{ ...camera, color_correction: false }]).wrist).toMatchObject({ color_correction: false });
  expect(serializeCameras([camera]).wrist).not.toHaveProperty("color_correction");
});
test("old profiles default to zero without changing capture dimensions", () => {
  expect(normalizeCameraConfig(camera)).toEqual({ ...camera, rotation: 0 });
});
test.each([0, 90, 180, 270] as const)("rotation %s survives profile and session serialization", (rotation) => {
  const saved = JSON.parse(JSON.stringify(normalizeCameraConfig({ ...camera, rotation })));
  expect(serializeCameras([saved]).wrist).toEqual({ type: "opencv", camera_id: "0x21100000c456367", width: 640, height: 480, fps: 30, rotation });
});
test.each([null, "90", -90, 45, 360, true, NaN])("invalid rotation %s is not replaced with zero", (value) => {
  expect(() => cameraRotation(value)).toThrow("Camera rotation must be");
});
test.each([0, 90, 180, 270] as const)("capture/output conversion is reversible at %s degrees", (rotation) => {
  const output = rotatedDimensions(640, 480, rotation);
  expect(output).toEqual(rotation === 90 || rotation === 270 ? { width: 480, height: 640 } : { width: 640, height: 480 });
  expect(rotatedDimensions(output.width, output.height, rotation)).toEqual({ width: 640, height: 480 });
});
test("rotation follows the camera id when a policy name is reassigned", () => {
  const saved = [{ ...camera, rotation: 180 as const }, { ...camera, id: "top", name: "top", camera_id: "0x1300000abd8050", rotation: 90 as const }];
  expect(rotationForCamera("0x21100000c456367", saved, {})).toBe(180);
  expect(rotationForCamera("0x1300000abd8050", saved, {})).toBe(90);
  expect(rotationForCamera("0x21100000c456367", saved, { "0x21100000c456367": 270 })).toBe(270);
  expect(rotationForCamera("unsaved", saved, {})).toBe(0);
});
test("a camera without an id has to be picked again", () => {
  expect(cameraConfigurationError([{ ...camera, camera_id: "" }])).toBe("wrist: pick the camera again");
  expect(cameraConfigurationError([camera])).toBeNull();
});
test("turning lens correction off is sent with the session and an omitted flag stays on", () => {
  expect(serializeCameras([{ ...camera, lens_correction: false }]).wrist).toMatchObject({ lens_correction: false });
  expect(serializeCameras([camera]).wrist).not.toHaveProperty("lens_correction");
});
test("preview and snapshot urls carry the capture settings and corrections", () => {
  const cam = { camera_id: "0x1300000abd8050", width: 640, height: 480, fps: 30, fourcc: "MJPG", lens_correction: false };
  const query = "?width=640&height=480&fps=30&fourcc=MJPG&lens=false&gain=true";
  expect(previewUrl("http://h", cam)).toBe(`http://h/camera-preview/0x1300000abd8050${query}`);
  expect(snapshotUrl("http://h", cam)).toBe(`http://h/camera-snapshot/0x1300000abd8050${query}`);
  expect(previewUrl("http://h", { ...cam, camera_id: "index:2" })).toContain("/camera-preview/index%3A2?");
});
