/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import '@tensorflow-models/face-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE, createDetector} from './shared/params';
import {setupStats} from './shared/stats_panel';
import {setBackendAndEnvFlags} from './shared/util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

function getIrisCenter(irisPoints) {
  let sumX = 0, sumY = 0;
  for (const p of irisPoints) {
    sumX += p.x;
    sumY += p.y;
  }
  return { x: sumX / irisPoints.length, y: sumY / irisPoints.length };
}


// 检测gaze方向
function detectGazeDirection(face) {
  const kps = face.keypoints;

  // 左眼 iris: 468(中心) + 469,470,471,472
  const leftIris = [kps[468], kps[469], kps[470], kps[471], kps[472]];
  const leftPupil = getIrisCenter(leftIris);

  // 右眼 iris: 473(中心) + 474,475,476,477
  const rightIris = [kps[473], kps[474], kps[475], kps[476], kps[477]];
  const rightPupil = getIrisCenter(rightIris);

  // 左眼边界
  const leftEye = {
    left: kps[33],
    right: kps[133],
    top: kps[159],
    bottom: kps[145],
    pupil: leftPupil,
    center: kps[468]
  };

  // 右眼边界
  const rightEye = {
    left: kps[362],
    right: kps[263],
    top: kps[386],
    bottom: kps[374],
    pupil: rightPupil,
    center: kps[473]
  };

  // 计算相对位置
  const relX = (
      (leftEye.pupil.x - leftEye.left.x) / (leftEye.right.x - leftEye.left.x) +
      (rightEye.pupil.x - rightEye.left.x) / (rightEye.right.x - rightEye.left.x)
  ) / 2;

  const relY = (
      (leftEye.pupil.y - leftEye.top.y) / (leftEye.bottom.y - leftEye.top.y) +
      (rightEye.pupil.y - rightEye.top.y) / (rightEye.bottom.y - rightEye.top.y)
  ) / 2;

  // 判断方向
  let gaze = "CENTER";
  if (relX < 0.35) gaze = "RIGHT";
  else if (relX > 0.65) gaze = "LEFT";
  else if (relY < 0.3 && relY > 0) {
    gaze = "UP";
  }
  else if (relY < 0) {
    gaze = "DOWN";
  }

  return gaze;
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateFaceStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateFaceStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let faces = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateFaces.
    beginEstimateFaceStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      faces =
          await detector.estimateFaces(camera.video, {flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateFaceStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (faces && faces.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(
        faces, STATE.modelConfig.triangulateMesh,
        STATE.modelConfig.boundingBox);

    for (const face of faces) {
      const gaze = detectGazeDirection(face);
      console.log("Gaze:", gaze);
    }
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  renderPrediction();
};

app();
