import { expect, test } from "vitest";
import { matchBrowserCameras, syncCameraIdentity, usbCameraId } from "./cameraMatch";

const browser = [
  { deviceId: "browser-a", label: "USB Camera (0abd:8050)" },
  { deviceId: "browser-b", label: "USB Camera (0abd:8050)" },
  { deviceId: "browser-innomaker", label: "Innomaker-U20CAM-720P" },
];

const backend = [
  { index: 1, name: "USB Camera", unique_id: "0x1300000abd8050", available: true },
  { index: 2, name: "USB Camera", unique_id: "0x1410000abd8050", available: true },
  { index: 3, name: "Innomaker-U20CAM-720P", unique_id: "0x21100000c456367", available: true },
];

test("a USB unique id is the same id lelab-camera list prints", () => {
  expect(usbCameraId("0x1300000abd8050")).toBe("0abd:8050@130000");
  expect(usbCameraId("0x11300000abd8050")).toBe("0abd:8050@1130000");
  expect(usbCameraId("6C707041-05AC-0011-0008-000000000001")).toBeNull();
});

test("two cameras with one name are not paired to a browser device by that name", () => {
  const matched = matchBrowserCameras(backend, browser);
  const twins = matched.filter((camera) => camera.name === "USB Camera");
  expect(twins.map((camera) => camera.deviceId)).toEqual(["", ""]);
  expect(twins[0].usbId).toBe("0abd:8050@130000");
  expect(twins[1].usbId).toBe("0abd:8050@141000");
  expect(twins[0].pictures?.map((picture) => picture.deviceId)).toEqual(["browser-a", "browser-b"]);
  expect(twins[1].pictures?.map((picture) => picture.deviceId)).toEqual(["browser-a", "browser-b"]);
  expect(matched[2]).toMatchObject({ deviceId: "browser-innomaker", usbId: "0c45:6367@2110000" });
  expect(matched[2].pictures).toBeUndefined();
});

test("a saved camera keeps the port it was given and does not follow the other camera's picture", () => {
  const available = matchBrowserCameras(backend, browser);
  const saved = { camera_index: 2, device_id: "browser-a", unique_id: undefined as string | undefined };
  const withPort = syncCameraIdentity(saved, available);
  expect(withPort).toMatchObject({ camera_index: 2, unique_id: "0x1410000abd8050", device_id: "browser-a" });
  const moved = syncCameraIdentity(
    { ...withPort, camera_index: 9 },
    available.map((camera) => (camera.index === 2 ? { ...camera, index: 4 } : camera)),
  );
  expect(moved.camera_index).toBe(4);
  expect(moved.device_id).toBe("browser-a");
});
