/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import * as mpHands from '@mediapipe/hands';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

// 计算三点是否近似共线
function arePointsColinear(p1, p2, p3, tolerance = 8e-2) {
  const v1 = [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z];
  const v2 = [p3.x - p1.x, p3.y - p1.y, p3.z - p1.z];
  const cross = [
    v1[1]*v2[2] - v1[2]*v2[1],
    v1[2]*v2[0] - v1[0]*v2[2],
    v1[0]*v2[1] - v1[1]*v2[0]
  ];
  const crossNorm = Math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2);
  const v1Norm = Math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2);
  const v2Norm = Math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2);

  if (v1Norm * v2Norm === 0) return false;
  const colinearMeasure = crossNorm / (v1Norm * v2Norm);
  return colinearMeasure < tolerance;
}

// 计算三点是否成“折角”
function arePointsTriangle(p1, p2, p3, tolerance = 15e-2) {
  const v1 = [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z];
  const v2 = [p3.x - p1.x, p3.y - p1.y, p3.z - p1.z];
  const cross = [
    v1[1]*v2[2] - v1[2]*v2[1],
    v1[2]*v2[0] - v1[0]*v2[2],
    v1[0]*v2[1] - v1[1]*v2[0]
  ];
  const crossNorm = Math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2);
  const v1Norm = Math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2);
  const v2Norm = Math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2);

  if (v1Norm * v2Norm === 0) return false;
  const colinearMeasure = crossNorm / (v1Norm * v2Norm);
  return colinearMeasure > tolerance;
}

// 竖中指
function checkMiddleFinger(landmarks) {
  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const middleTip = landmarks[12];
  const middlePip = landmarks[10];
  const ringTip = landmarks[16];
  const ringPip = landmarks[14];
  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];

  if (middleTip.y < middlePip.y &&
      indexTip.y > indexPip.y &&
      ringTip.y > ringPip.y &&
      pinkyTip.y > pinkyPip.y) {
    return "Middle Finger";
  }
  return null;
}

// 拇指松散指向
function checkFingerPointing(landmarks) {
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const thumbMcp = landmarks[2];
  const indexTip = landmarks[8];
  const indexDip = landmarks[7];
  const indexPip = landmarks[6];
  const middleTip = landmarks[12];
  const middleDip = landmarks[11];
  const middlePip = landmarks[10];
  const ringTip = landmarks[16];
  const ringDip = landmarks[15];
  const ringPip = landmarks[14];

  const middleRingTriangle =
      arePointsTriangle(middleTip, middleDip, middlePip) &&
      arePointsTriangle(ringTip, ringDip, ringPip);

  const thumbIndexColinear =
      arePointsColinear(thumbTip, thumbIp, thumbMcp) ||
      arePointsColinear(indexTip, indexDip, indexPip);

  if (middleRingTriangle && thumbIndexColinear) {
    return "Pointing";
  }
  return null;
}

// 手心朝上
function checkPalmUpward(landmarks) {
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  const middleMcp = landmarks[9];
  const pinkyMcp = landmarks[17];
  const indexTip = landmarks[8];
  const indexDip = landmarks[7];
  const indexPip = landmarks[6];
  const middleTip = landmarks[12];
  const middleDip = landmarks[11];
  const middlePip = landmarks[10];

  function angleWithHorizon(p1, p2, thresh=70) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const angleDeg = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
    return angleDeg < thresh || angleDeg > (180 - thresh);
  }

  if (angleWithHorizon(wrist, indexMcp) &&
      angleWithHorizon(wrist, middleMcp) &&
      angleWithHorizon(wrist, pinkyMcp) &&
      arePointsColinear(indexTip, indexDip, indexPip, 0.3) &&
      arePointsColinear(middleTip, middleDip, middlePip, 0.3)) {
    return "Palm Upward";
  }
  return null;
}

// 判断两条线是否平行（2D）
function areLinesParallel2D(p1, p2, q1, q2, toleranceDeg = 20) {
  const v1 = [p2.x - p1.x, p2.y - p1.y];
  const v2 = [q2.x - q1.x, q2.y - q1.y];

  const dot = v1[0]*v2[0] + v1[1]*v2[1];
  const norm1 = Math.sqrt(v1[0]**2 + v1[1]**2);
  const norm2 = Math.sqrt(v2[0]**2 + v2[1]**2);
  if (norm1 * norm2 === 0) return false;

  const cosTheta = dot / (norm1 * norm2);
  const angleDeg = Math.acos(Math.min(1, Math.max(-1, cosTheta))) * 180 / Math.PI;

  return angleDeg < toleranceDeg || Math.abs(180 - angleDeg) < toleranceDeg;
}

// 判断三点是否近似共线（2D）
function arePointsColinear2D(p1, p2, p3, tolerance = 0.1) {
  const v1 = [p2.x - p1.x, p2.y - p1.y];
  const v2 = [p3.x - p1.x, p3.y - p1.y];
  const cross = v1[0] * v2[1] - v1[1] * v2[0]; // 2D 叉积
  const area = Math.abs(cross);
  const v1Len = Math.sqrt(v1[0] ** 2 + v1[1] ** 2);
  const v2Len = Math.sqrt(v2[0] ** 2 + v2[1] ** 2);
  if (v1Len * v2Len === 0) return false;
  const colinearMeasure = area / (v1Len * v2Len);
  return colinearMeasure < tolerance;
}

// 判断一根手指是否伸直
function isFingerStraight(hand, mcpIdx, pipIdx, dipIdx, tipIdx, tolerance = 0.15) {
  const mcp = hand[mcpIdx];
  const pip = hand[pipIdx];
  const dip = hand[dipIdx];
  const tip = hand[tipIdx];
  // 简单判定：MCP, PIP, TIP 基本共线，且 DIP 也落在直线上
  return arePointsColinear2D(mcp, pip, tip, tolerance) &&
      arePointsColinear2D(pip, dip, tip, tolerance);
}

// 判断单手是否张开（大部分手指伸直）
function isHandOpen(hand) {
  if (!hand) return false;

  let straightCount = 0;
  if (isFingerStraight(hand, 5, 6, 7, 8)) straightCount++;   // index
  if (isFingerStraight(hand, 9, 10, 11, 12)) straightCount++; // middle
  if (isFingerStraight(hand, 13, 14, 15, 16)) straightCount++; // ring
  if (isFingerStraight(hand, 17, 18, 19, 20)) straightCount++; // pinky

  // 至少 3 根手指伸直，认为是 open hand
  return straightCount >= 3;
}

function checkOpenPalm(leftHand, rightHand) {
  if (!leftHand || !rightHand) return null;

  // Step 1: 单手张开
  if (!isHandOpen(leftHand) || !isHandOpen(rightHand)) return null;

  // Step 2: 两只手整体平行（用中指或食指）
  const leftWrist = leftHand[0];
  const rightWrist = rightHand[0];

  const leftMiddle = leftHand[12];
  const rightMiddle = rightHand[12];

  if (areLinesParallel2D(leftWrist, leftMiddle, rightWrist, rightMiddle, 20)) {
    return "Open Palm";
  }
  return null;
}

function distance2D(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx*dx + dy*dy);
}

function checkFingerInterlocked(leftHand, rightHand) {
  if (!leftHand || !rightHand) return null;

  // 取 [3,7,11,15,19] 这几个关节
  const jointIdx = [3, 6, 10, 14, 18];
  const leftJoints  = jointIdx.map(i => leftHand[i]);
  const rightJoints = jointIdx.map(i => rightHand[i]);

  let interlockedCount = 0;
  const threshold = 40;

  for (let i = 0; i < leftJoints.length; i++) {
    const lf = leftJoints[i];
    const rf = rightJoints[i];
    const d = distance2D(lf, rf);
    if (d < threshold) {
      interlockedCount++;
    }
  }

  if (interlockedCount >= 3) {
    return "Finger Interlocked";
  }
  return null;
}

// 总入口
function detectGesture(hands) {
  if (!hands || hands.length === 0) return null;

  // 1. 双手规则：至少两只手时检测
  if (hands.length >= 2) {
    const left = hands.find(h => h.handedness === "Left");
    const right = hands.find(h => h.handedness === "Right");

    if (left && right) {
      const leftLm3D = left.keypoints3D;
      const rightLm3D = right.keypoints3D;

      const leftLm = left.keypoints;
      const rightLm = right.keypoints;

      if (leftLm && rightLm) {
        let g = checkFingerInterlocked(leftLm, rightLm);
        if (g) return g;

        g = checkOpenPalm(leftLm, rightLm);
        if (g) return g;
      }
    }
  }

  // 2. 单手规则：循环每只手检测
  for (const hand of hands) {
    const landmarks = hand.keypoints;
    if (!landmarks) continue;

    let g = checkMiddleFinger(landmarks);
    if (g) return g;

    // g = checkFingerPointing(landmarks);
    // if (g) return g;

    // g = checkPalmUpward(landmarks);
    // if (g) return g;
  }

  return null;
}
tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as handdetection from '@tensorflow-models/hand-pose-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './shared/params';
import {setupStats} from './shared/stats_panel';
import {setBackendAndEnvFlags} from './shared/util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

async function createDetector() {
  switch (STATE.model) {
    case handdetection.SupportedModels.MediaPipeHands:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return handdetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          maxHands: STATE.modelConfig.maxNumHands
        });
      }
  }
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

function beginEstimateHandsStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateHandsStats() {
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

  let hands = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateHands.
    beginEstimateHandsStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      hands = await detector.estimateHands(
          camera.video,
          {flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateHandsStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (hands && hands.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(hands);
    const gesture = detectGesture(hands);
    if (gesture) {
      console.log("Gesture:", gesture);
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
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  renderPrediction();
};

app();
