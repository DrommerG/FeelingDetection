import * as ort from 'onnxruntime-web';
import wasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import wasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import wasmAsyncifyModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import wasmAsyncifyBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { createIcons, icons } from 'lucide';
import './styles.css';

createIcons({ icons });

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { mjs: wasmModuleUrl, wasm: wasmBinaryUrl };

const palette = ['#b9f25a', '#ff785e', '#7ca8ff', '#f3d54f', '#d69bff', '#52dbc0', '#ff9ccb'];
const expressions = {
	sad: 'Tristeza',
	disgust: 'Incomodidad',
	angry: 'Enojo',
	neutral: 'Neutral',
	fear: 'Miedo',
	surprise: 'Sorpresa',
	happy: 'Felicidad',
};
const expressionLabels = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy'];
const detectionInterval = 240;

const elements = {
	cameraSelect: document.querySelector('#cameraSelect'),
	cameraStage: document.querySelector('#cameraStage'),
	coordinateList: document.querySelector('#coordinateList'),
	emptyState: document.querySelector('#emptyState'),
	emotionCount: document.querySelector('#emotionCount'),
	emotionLabels: document.querySelector('#emotionLabels'),
	faceCount: document.querySelector('#faceCount'),
	faceList: document.querySelector('#faceList'),
	faceTotal: document.querySelector('#faceTotal'),
	flipButton: document.querySelector('#flipButton'),
	frameSize: document.querySelector('#frameSize'),
	fullscreenButton: document.querySelector('#fullscreenButton'),
	livePill: document.querySelector('#livePill'),
	overlay: document.querySelector('#overlay'),
	stageError: document.querySelector('#stageError'),
	startButton: document.querySelector('#startButton'),
	statusIndicator: document.querySelector('#statusIndicator'),
	statusText: document.querySelector('#statusText'),
	stopButton: document.querySelector('#stopButton'),
	thresholdSlider: document.querySelector('#thresholdSlider'),
	thresholdValue: document.querySelector('#thresholdValue'),
	video: document.querySelector('#video'),
	reflectionForm: document.querySelector('#reflectionForm'),
	reflectionText: document.querySelector('#reflectionText'),
	reflectionResponse: document.querySelector('#reflectionResponse'),
};

const context = elements.overlay.getContext('2d');
const cropCanvas = document.createElement('canvas');
const cropContext = cropCanvas.getContext('2d', { willReadFrequently: true });
let cameraStream = null;
let detectorSession = null;
let emotionSession = null;
let detectorBackend = 'WASM';
let emotionBackend = 'WASM';
let modelLoadPromise = null;
let detectionTimer = null;
let renderFrame = null;
let detectorBusy = false;
let emotionBusy = false;
let mirrored = false;
let nextTrackId = 1;
let cameraGeneration = 0;
let tracks = [];
let visibleFaces = [];
const emotionQueue = [];

function setStatus(message, state = 'idle') {
	elements.statusText.textContent = message;
	elements.statusIndicator.className = `status-indicator ${state}`;
}

function setLiveState(message, active = false) {
	elements.livePill.innerHTML = `<span></span>${message}`;
	elements.livePill.classList.toggle('is-live', active);
}

function showStageError(message) {
	elements.stageError.textContent = message;
	elements.stageError.hidden = false;
}

function clearStageError() {
	elements.stageError.hidden = true;
	elements.stageError.textContent = '';
}

async function fetchModel(name) {
	const response = await fetch(`${import.meta.env.BASE_URL}models/${name}.onnx`, { cache: 'force-cache' });
	if (!response.ok || !response.body) {
		throw new Error(`No se pudo descargar el modelo ${name}.`);
	}

	const total = Number(response.headers.get('content-length')) || 0;
	const reader = response.body.getReader();
	const chunks = [];
	let loaded = 0;

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		loaded += value.length;
		const size = (loaded / 1024 / 1024).toFixed(1);
		const totalLabel = total ? ` / ${(total / 1024 / 1024).toFixed(0)} MB` : ' MB';
		setLiveState(`CARGANDO ${name === 'face' ? 'YOLO' : 'FER'} · ${size}${totalLabel}`);
		setStatus(`Descargando modelo ${name === 'face' ? 'YOLO' : 'de expresiones'}`, 'loading');
	}

	const bytes = new Uint8Array(loaded);
	let offset = 0;
	for (const chunk of chunks) {
		bytes.set(chunk, offset);
		offset += chunk.length;
	}
	return bytes;
}


function errorMessage(error) {
	return error instanceof Error ? error.message : String(error);
}

async function createModelSession(bytes, modelName, useWebGPU = true) {
	const backendErrors = [];
	if (useWebGPU && navigator.gpu) {
		try {
			const gpuRuntime = await import('onnxruntime-web/webgpu');
			gpuRuntime.env.wasm.numThreads = 1;
			gpuRuntime.env.wasm.wasmPaths = { mjs: wasmAsyncifyModuleUrl, wasm: wasmAsyncifyBinaryUrl };
			const session = await gpuRuntime.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
			return { session, backend: 'WebGPU' };
		} catch (error) {
			backendErrors.push(`WebGPU: ${errorMessage(error)}`);
			console.info('WebGPU no disponible para este modelo; se usará WASM.', error);
		}
	}

	try {
		const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
		return { session, backend: 'WASM' };
	} catch (error) {
		backendErrors.push(`WASM: ${errorMessage(error)}`);
		console.error(`No se pudo iniciar ${modelName}:`, error);
		throw new Error(`No se pudo iniciar ${modelName}. ${backendErrors.join(' | ')}`);
	}
}

async function loadModels() {
	if (modelLoadPromise) return modelLoadPromise;

	modelLoadPromise = (async () => {
		setLiveState('CARGANDO YOLO');
		const faceBytes = await fetchModel('face');
		const detector = await createModelSession(faceBytes, 'el detector facial YOLO', false);
		detectorSession = detector.session;
		detectorBackend = detector.backend;
		setLiveState('CARGANDO CLASIFICADOR');
		const emotionBytes = await fetchModel('emotion');
		const classifier = await createModelSession(emotionBytes, 'el clasificador de expresiones');
		emotionSession = classifier.session;
		emotionBackend = classifier.backend;
		const backendLabel = detectorBackend === 'WebGPU' && emotionBackend === 'WebGPU' ? 'GPU' : 'WASM';
		setStatus(`Análisis local activo · ${backendLabel}`, 'ready');
		setLiveState(`ANÁLISIS ACTIVO · ${backendLabel}`, true);
	})();

	try {
		await modelLoadPromise;
	} catch (error) {
		modelLoadPromise = null;
		throw error;
	}
}

async function refreshCameraOptions() {
	const devices = await navigator.mediaDevices.enumerateDevices();
	const cameras = devices.filter((device) => device.kind === 'videoinput');
	const activeDevice = cameraStream?.getVideoTracks()[0]?.getSettings().deviceId;
	elements.cameraSelect.replaceChildren();

	cameras.forEach((camera, index) => {
		const option = document.createElement('option');
		option.value = camera.deviceId;
		option.textContent = camera.label || `Cámara ${index + 1}`;
		option.selected = camera.deviceId === activeDevice;
		elements.cameraSelect.append(option);
	});
	elements.cameraSelect.disabled = cameras.length < 2;
}

async function openCamera(deviceId = '') {
	if (!navigator.mediaDevices?.getUserMedia) {
		throw new Error('Esta página necesita HTTPS o localhost y un navegador con acceso a cámara.');
	}

	const generation = ++cameraGeneration;
	clearTimeout(detectionTimer);
	cancelAnimationFrame(renderFrame);
	visibleFaces = [];
	tracks = [];
	emotionQueue.length = 0;
	elements.faceCount.textContent = '00';
	elements.faceTotal.textContent = '00';
	renderAnalysis();
	if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
	cameraStream = null;
	elements.video.srcObject = null;
	const video = deviceId
		? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
		: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } };
	const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
	if (generation !== cameraGeneration) {
		stream.getTracks().forEach((track) => track.stop());
		return;
	}
	cameraStream = stream;
	elements.video.srcObject = stream;
	await elements.video.play();
	await new Promise((resolve) => {
		if (elements.video.videoWidth) resolve();
		else elements.video.addEventListener('loadedmetadata', resolve, { once: true });
	});

	elements.cameraStage.style.aspectRatio = `${elements.video.videoWidth} / ${elements.video.videoHeight}`;
	elements.frameSize.textContent = `${elements.video.videoWidth} × ${elements.video.videoHeight}`;
	elements.emptyState.hidden = true;
	elements.stopButton.hidden = false;
	elements.flipButton.disabled = false;
	elements.cameraSelect.disabled = false;
	clearStageError();
	setStatus('Conectando con la cámara', 'loading');
	setLiveState('INICIANDO');
	await refreshCameraOptions();
	renderLoop(generation);

	try {
		await loadModels();
		if (generation === cameraGeneration) runDetectionLoop(generation);
	} catch (error) {
		if (generation !== cameraGeneration) return;
		setStatus('No se pudo iniciar el análisis', 'error');
		setLiveState('MODELO NO DISPONIBLE');
		showStageError(`${error.message} Comprueba tu conexión e inténtalo de nuevo.`);
		console.error(error);
	}
}

function stopCamera() {
	cameraGeneration += 1;
	if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
	cameraStream = null;
	elements.video.srcObject = null;
	clearTimeout(detectionTimer);
	cancelAnimationFrame(renderFrame);
	context.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
	elements.emptyState.hidden = false;
	elements.stopButton.hidden = true;
	elements.flipButton.disabled = true;
	elements.cameraSelect.disabled = true;
	elements.frameSize.textContent = '-- × --';
	elements.faceCount.textContent = '00';
	elements.faceTotal.textContent = '00';
	elements.cameraStage.style.aspectRatio = '16 / 9';
	visibleFaces = [];
	tracks = [];
	emotionQueue.length = 0;
	renderAnalysis();
	setStatus('Cámara desactivada');
	setLiveState('EN ESPERA');
}

function letterboxFrame(video, width, height) {
	const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
	const resizedWidth = Math.round(video.videoWidth * scale);
	const resizedHeight = Math.round(video.videoHeight * scale);
	const padX = Math.floor((width - resizedWidth) / 2);
	const padY = Math.floor((height - resizedHeight) / 2);

	cropCanvas.width = width;
	cropCanvas.height = height;
	cropContext.fillStyle = 'rgb(114, 114, 114)';
	cropContext.fillRect(0, 0, width, height);
	cropContext.drawImage(video, padX, padY, resizedWidth, resizedHeight);
	const { data } = cropContext.getImageData(0, 0, width, height);
	const tensorData = new Float32Array(3 * width * height);
	const planeSize = width * height;

	for (let pixel = 0; pixel < planeSize; pixel += 1) {
		const source = pixel * 4;
		tensorData[pixel] = data[source] / 255;
		tensorData[planeSize + pixel] = data[source + 1] / 255;
		tensorData[planeSize * 2 + pixel] = data[source + 2] / 255;
	}

	return {
		tensor: new ort.Tensor('float32', tensorData, [1, 3, height, width]),
		scale,
		padX,
		padY,
		width,
		height,
	};
}

function intersectionOverUnion(first, second) {
	const left = Math.max(first.x1, second.x1);
	const top = Math.max(first.y1, second.y1);
	const right = Math.min(first.x2, second.x2);
	const bottom = Math.min(first.y2, second.y2);
	const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
	const areaFirst = (first.x2 - first.x1) * (first.y2 - first.y1);
	const areaSecond = (second.x2 - second.x1) * (second.y2 - second.y1);
	return intersection / Math.max(areaFirst + areaSecond - intersection, 1);
}

function suppressOverlaps(detections, threshold = 0.48) {
	const sorted = detections.sort((first, second) => second.score - first.score);
	const kept = [];
	while (sorted.length && kept.length < 30) {
		const candidate = sorted.shift();
		kept.push(candidate);
		for (let index = sorted.length - 1; index >= 0; index -= 1) {
			if (intersectionOverUnion(candidate, sorted[index]) > threshold) sorted.splice(index, 1);
		}
	}
	return kept;
}

function decodeDetections(output, frame, threshold) {
	const dimensions = output.dims;
	const values = output.data;
	const count = dimensions.length >= 3 ? dimensions.at(-2) : 0;
	const features = dimensions.length >= 3 ? dimensions.at(-1) : 0;
	const detections = [];

	if (features >= 6 && features <= 8) {
		for (let index = 0; index < count; index += 1) {
			const offset = index * features;
			const score = values[offset + 4];
			if (score < threshold) continue;
			detections.push({
				x1: values[offset], y1: values[offset + 1],
				x2: values[offset + 2], y2: values[offset + 3], score,
			});
		}
	} else if (dimensions.length === 3 && dimensions[1] >= 5 && dimensions[1] <= 8) {
		const boxes = dimensions[2];
		for (let index = 0; index < boxes; index += 1) {
			const score = values[4 * boxes + index];
			if (score < threshold) continue;
			const centerX = values[index];
			const centerY = values[boxes + index];
			const width = values[boxes * 2 + index];
			const height = values[boxes * 3 + index];
			detections.push({
				x1: centerX - width / 2, y1: centerY - height / 2,
				x2: centerX + width / 2, y2: centerY + height / 2,
				score,
			});
		}
	}

	const frameWidth = elements.video.videoWidth;
	const frameHeight = elements.video.videoHeight;
	const normalized = detections.map((detection) => {
		const coordinatesAreNormalized = Math.max(Math.abs(detection.x2), Math.abs(detection.y2)) <= 1.5;
		const modelScaleX = coordinatesAreNormalized ? frame.width : 1;
		const modelScaleY = coordinatesAreNormalized ? frame.height : 1;
		return {
			x1: Math.max(0, Math.min(frameWidth, (detection.x1 * modelScaleX - frame.padX) / frame.scale)),
			y1: Math.max(0, Math.min(frameHeight, (detection.y1 * modelScaleY - frame.padY) / frame.scale)),
			x2: Math.max(0, Math.min(frameWidth, (detection.x2 * modelScaleX - frame.padX) / frame.scale)),
			y2: Math.max(0, Math.min(frameHeight, (detection.y2 * modelScaleY - frame.padY) / frame.scale)),
			score: detection.score,
		};
	}).filter((box) => box.x2 - box.x1 >= 8 && box.y2 - box.y1 >= 8);

	return suppressOverlaps(normalized);
}

function matchTracks(detections) {
	const now = performance.now();
	const assigned = new Set();
	const current = detections.map((detection) => {
		let match = null;
		let bestScore = 0.18;

		for (const track of tracks) {
			if (assigned.has(track.id) || now - track.lastSeen > 1300) continue;
			const overlap = intersectionOverUnion(detection, track.box);
			if (overlap > bestScore) {
				match = track;
				bestScore = overlap;
			}
		}

		if (!match) {
			match = {
				id: nextTrackId++,
				emotion: null,
				lastEmotionAt: 0,
				emotionQueued: false,
			};
		}

		Object.assign(match, { box: detection, lastSeen: now });
		assigned.add(match.id);
		return match;
	});

	tracks = [...tracks.filter((track) => now - track.lastSeen <= 1300), ...current.filter((track) => !tracks.includes(track))];
	return current;
}

function cropFaceTensor(face) {
	const box = face.box;
	const width = box.x2 - box.x1;
	const height = box.y2 - box.y1;
	const side = Math.max(width, height) * 1.2;
	const centerX = (box.x1 + box.x2) / 2;
	const centerY = (box.y1 + box.y2) / 2;
	cropCanvas.width = 224;
	cropCanvas.height = 224;
	cropContext.drawImage(
		elements.video,
		centerX - side / 2, centerY - side / 2, side, side,
		0, 0, 224, 224,
	);

	const { data } = cropContext.getImageData(0, 0, 224, 224);
	const planeSize = 224 * 224;
	const tensorData = new Float32Array(planeSize * 3);
	for (let pixel = 0; pixel < planeSize; pixel += 1) {
		const source = pixel * 4;
		tensorData[pixel] = data[source] / 127.5 - 1;
		tensorData[planeSize + pixel] = data[source + 1] / 127.5 - 1;
		tensorData[planeSize * 2 + pixel] = data[source + 2] / 127.5 - 1;
	}
	return new ort.Tensor('float32', tensorData, [1, 3, 224, 224]);
}

function scheduleEmotionReads(faces) {
	const now = performance.now();
	for (const face of faces) {
		if (face.emotionQueued || now - face.lastEmotionAt < 1700) continue;
		face.lastEmotionAt = now;
		face.emotionQueued = true;
		emotionQueue.push({ trackId: face.id, tensor: cropFaceTensor(face) });
	}
	processEmotionQueue();
}

async function processEmotionQueue() {
	if (emotionBusy || !emotionSession) return;
	emotionBusy = true;
	while (emotionQueue.length) {
		const { trackId, tensor } = emotionQueue.shift();
		const track = tracks.find((item) => item.id === trackId);
		if (!track) continue;

		try {
			const result = await emotionSession.run({ [emotionSession.inputNames[0]]: tensor });
			const logits = result[emotionSession.outputNames[0]].data;
			const max = Math.max(...logits);
			const exponents = Array.from(logits, (value) => Math.exp(value - max));
			const total = exponents.reduce((sum, value) => sum + value, 0);
			const ranked = exponents.map((value, index) => ({ key: expressionLabels[index], score: value / total }));
			ranked.sort((first, second) => second.score - first.score);

			const currentTrack = tracks.find((item) => item.id === trackId);
			if (currentTrack) currentTrack.emotion = ranked[0];
			renderAnalysis();
		} catch (error) {
			console.error('Expression classification error:', error);
			setStatus('Error al clasificar expresiones', 'error');
		} finally {
			const currentTrack = tracks.find((item) => item.id === trackId);
			if (currentTrack) currentTrack.emotionQueued = false;
		}
	}
	emotionBusy = false;
}

async function runDetectionLoop(generation = cameraGeneration) {
	if (generation !== cameraGeneration || !cameraStream || !detectorSession) return;
	if (detectorBusy) {
		detectionTimer = window.setTimeout(() => runDetectionLoop(generation), 90);
		return;
	}
	detectorBusy = true;
	const startedAt = performance.now();

	try {
		const metadata = detectorSession.inputMetadata?.find((item) => item.name === detectorSession.inputNames[0]);
		const dimensions = metadata?.shape || [1, 3, 640, 640];
		const inputHeight = Number(dimensions.at(-2)) || 640;
		const inputWidth = Number(dimensions.at(-1)) || 640;
		elements.overlay.width = elements.video.videoWidth;
		elements.overlay.height = elements.video.videoHeight;
		const frame = letterboxFrame(elements.video, inputWidth, inputHeight);
		const result = await detectorSession.run({ [detectorSession.inputNames[0]]: frame.tensor });
		if (generation !== cameraGeneration || !cameraStream) return;
		const rawOutput = result[detectorSession.outputNames[0]];
		const decoded = decodeDetections(rawOutput, frame, Number(elements.thresholdSlider.value) / 100);
		visibleFaces = matchTracks(decoded);
		elements.faceCount.textContent = String(visibleFaces.length).padStart(2, '0');
		elements.faceTotal.textContent = String(visibleFaces.length).padStart(2, '0');
		renderAnalysis();
		scheduleEmotionReads(visibleFaces);
	} catch (error) {
		console.error('Face detection error:', error);
		setStatus('Error en la detección YOLO', 'error');
		setLiveState('ERROR DE DETECCIÓN');
		showStageError(`No se pudo ejecutar YOLO con ${detectorBackend}: ${errorMessage(error)}`);
		return;
	} finally {
		detectorBusy = false;
	}

	const elapsed = performance.now() - startedAt;
	if (generation === cameraGeneration && cameraStream) {
		detectionTimer = window.setTimeout(() => runDetectionLoop(generation), Math.max(90, detectionInterval - elapsed));
	}
}

function renderLoop(generation = cameraGeneration) {
	if (generation !== cameraGeneration || !cameraStream) return;
	const width = elements.video.videoWidth;
	const height = elements.video.videoHeight;
	if (width && height) {
		if (elements.overlay.width !== width || elements.overlay.height !== height) {
			elements.overlay.width = width;
			elements.overlay.height = height;
		}
		context.clearRect(0, 0, width, height);
		context.lineWidth = Math.max(2, width / 500);
		context.font = `600 ${Math.max(13, Math.round(width / 72))}px Manrope, sans-serif`;
		context.textBaseline = 'middle';

		for (const face of visibleFaces) {
			const color = palette[(face.id - 1) % palette.length];
			const { x1, y1, x2, y2 } = face.box;
			const boxWidth = x2 - x1;
			const boxHeight = y2 - y1;
			context.strokeStyle = color;
			context.strokeRect(x1, y1, boxWidth, boxHeight);

			const label = face.emotion
				? `ROSTRO ${String(face.id).padStart(2, '0')} · ${expressions[face.emotion.key].toUpperCase()} ${Math.round(face.emotion.score * 100)}%`
				: `ROSTRO ${String(face.id).padStart(2, '0')} · ANALIZANDO`;
			const labelHeight = Math.max(26, Math.round(width / 37));
			const labelWidth = Math.min(width - 4, context.measureText(label).width + labelHeight * 0.9);
			const labelX = Math.max(0, Math.min(width - labelWidth, x1));
			const labelY = y2 + labelHeight < height ? y2 : Math.max(0, y1 - labelHeight);
			context.fillStyle = color;
			context.fillRect(labelX, labelY, labelWidth, labelHeight);
			context.fillStyle = '#17211c';
			context.fillText(label, labelX + labelHeight * 0.42, labelY + labelHeight / 2, labelWidth - labelHeight * 0.7);
		}
	}
	renderFrame = requestAnimationFrame(() => renderLoop(generation));
}

function displayBox(box) {
	const x = mirrored ? elements.video.videoWidth - box.x2 : box.x1;
	return {
		x: Math.round(x),
		y: Math.round(box.y1),
		width: Math.round(box.x2 - box.x1),
		height: Math.round(box.y2 - box.y1),
	};
}

function renderAnalysis() {
	if (!visibleFaces.length) {
		elements.faceList.innerHTML = '<span class="insight-kicker">UN MOMENTO PARA TI</span><p>Cuando haya una lectura, puedes tomarla como una invitación a hacer una pausa, no como una conclusión.</p>';
		elements.coordinateList.innerHTML = '<p class="coordinate-empty">Esperando detecciones <span>·</span> píxeles</p>';
		elements.emotionLabels.innerHTML = `<span class="quiet-placeholder">${cameraStream ? 'No hay rostros visibles en el cuadro.' : 'Las expresiones estimadas aparecerán aquí.'}</span>`;
		elements.emotionCount.textContent = cameraStream ? 'Buscando un rostro…' : 'Sin lecturas todavía';
		return;
	}

	const firstEmotion = visibleFaces.find((face) => face.emotion)?.emotion;
	const insight = firstEmotion
		? `La expresión visible parece ${expressions[firstEmotion.key].toLowerCase()}. Haz una pausa y nota cómo te sientes de verdad.`
		: 'Veo un rostro. Estoy observando su expresión…';
	elements.faceList.innerHTML = `<span class="insight-kicker">UNA LECTURA SUAVE</span><p>${insight}</p>`;

	elements.coordinateList.innerHTML = visibleFaces.map((face) => {
		const color = palette[(face.id - 1) % palette.length];
		const box = displayBox(face.box);
		return `<div class="coordinate-row">
			<span class="coordinate-marker" style="--face-color:${color}"></span>
			<span class="coordinate-name">R${String(face.id).padStart(2, '0')}</span>
			<span class="coordinate-values">X ${box.x} <i></i> Y ${box.y} <i></i> W ${box.width} <i></i> H ${box.height}</span>
		</div>`;
	}).join('');

	elements.emotionLabels.innerHTML = visibleFaces.map((face) => {
		const color = palette[(face.id - 1) % palette.length];
		const text = face.emotion ? expressions[face.emotion.key] : 'Analizando';
		const score = face.emotion ? `${Math.round(face.emotion.score * 100)}%` : '···';
		return `<span class="emotion-chip" style="--face-color:${color}"><span class="emotion-chip-dot"></span><span>R${String(face.id).padStart(2, '0')}</span><strong>${text}</strong><small>${score}</small></span>`;
	}).join('');
	elements.emotionCount.textContent = `${visibleFaces.length} ${visibleFaces.length === 1 ? 'ROSTRO' : 'ROSTROS'}`;
}

elements.startButton.addEventListener('click', async () => {
	elements.startButton.disabled = true;
	elements.startButton.querySelector('span').textContent = 'Solicitando acceso…';
	try {
		await openCamera();
	} catch (error) {
		setStatus('No se pudo acceder a la cámara', 'error');
		setLiveState('CÁMARA NO DISPONIBLE');
		showStageError(error.name === 'NotAllowedError'
			? 'Permite el acceso a la cámara en el navegador para iniciar.'
			: error.message || 'No se encontró una cámara disponible.');
	} finally {
		elements.startButton.disabled = false;
		elements.startButton.querySelector('span').textContent = 'Activar cámara';
	}
});

elements.stopButton.addEventListener('click', stopCamera);

elements.cameraSelect.addEventListener('change', async (event) => {
	try {
		await openCamera(event.target.value);
	} catch (error) {
		setStatus('No se pudo cambiar de cámara', 'error');
		showStageError(error.message || 'No se pudo abrir la cámara seleccionada.');
	}
});

elements.flipButton.addEventListener('click', () => {
	mirrored = !mirrored;
	elements.cameraStage.classList.toggle('is-mirrored', mirrored);
	elements.flipButton.setAttribute('aria-pressed', String(mirrored));
	renderAnalysis();
});

elements.fullscreenButton.addEventListener('click', async () => {
	try {
		if (document.fullscreenElement) await document.exitFullscreen();
		else await elements.cameraStage.requestFullscreen();
	} catch {
		setStatus('Pantalla completa no disponible', 'error');
	}
});

elements.thresholdSlider.addEventListener('input', (event) => {
	elements.thresholdValue.textContent = `${event.target.value}%`;
});

elements.reflectionForm.addEventListener('submit', (event) => {
	event.preventDefault();
	const hasWords = elements.reflectionText.value.trim().length > 0;
	elements.reflectionResponse.textContent = hasWords
		? 'Gracias por ponerlo en palabras. Respira un momento y piensa qué necesitas ahora.'
		: 'Puedes escribir un poco o simplemente tomarte un momento para hacer una pausa.';
	elements.reflectionResponse.hidden = false;
});

window.addEventListener('pagehide', () => {
	if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
});