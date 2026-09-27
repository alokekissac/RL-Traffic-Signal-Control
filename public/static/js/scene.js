// 3D intersection built with three.js. Purely a view: app.js owns the simulation
// and tells the scene what happened each tick.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const ROAD_HALF = 8;          // road is 16 wide (two lanes)
const STOP = 13.5;            // stop-line distance from the centre
const SLOT0 = STOP + 3.2;     // first queued car centre
const GAP = 6.0;              // spacing between queued cars
const LANE = 4;               // lane centre offset
const SPAWN = 115, EXIT = 150;

// NS traffic drives north (-z) in lane x=+4; EW traffic drives east (+x) in lane z=+4
const nsSlot = (i) => new THREE.Vector3(LANE, 0, SLOT0 + i * GAP);
const ewSlot = (i) => new THREE.Vector3(-(SLOT0 + i * GAP), 0, LANE);
const ROUTES = {
  ns: { slot: nsSlot, spawn: new THREE.Vector3(LANE, 0, SPAWN), exit: new THREE.Vector3(LANE, 0, -EXIT), back: new THREE.Vector3(0, 0, 1), yaw: 0 },
  ew: { slot: ewSlot, spawn: new THREE.Vector3(-SPAWN, 0, LANE), exit: new THREE.Vector3(EXIT, 0, LANE), back: new THREE.Vector3(-1, 0, 0), yaw: -Math.PI / 2 },
};

const ease = {
  linear: (t) => t,
  inQuad: (t) => t * t,
  outCubic: (t) => 1 - Math.pow(1 - t, 3),
  inOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
};

const CAR_COLORS = ["#eef1f5", "#4b5563", "#d62f2f", "#2f6fe8", "#19a7e8", "#f2a516", "#16b07a", "#b6bdc6", "#8b5cf6", "#e36a2e", "#f3f4f6", "#c7d2fe"];

// ---------------------------------------------------------------- textures
function canvasTexture(w, h, draw) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function skyTexture() {
  return canvasTexture(16, 512, (g, w, h) => {
    const grad = g.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "#0a1426");
    grad.addColorStop(0.45, "#1a2f55");
    grad.addColorStop(0.62, "#3b4f7a");
    grad.addColorStop(0.72, "#b56f5a");
    grad.addColorStop(0.8, "#e59a64");
    grad.addColorStop(1, "#2a2f40");
    g.fillStyle = grad; g.fillRect(0, 0, w, h);
  });
}

function asphaltTexture() {
  const t = canvasTexture(256, 256, (g, w, h) => {
    g.fillStyle = "#30343c"; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 5000; i++) {
      const v = 38 + Math.random() * 28;
      g.fillStyle = `rgba(${v},${v + 2},${v + 6},${0.35 + Math.random() * 0.4})`;
      g.fillRect(Math.random() * w, Math.random() * h, 1.4, 1.4);
    }
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function windowTexture(seed) {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const t = canvasTexture(128, 256, (g, w, h) => {
    g.fillStyle = "#000"; g.fillRect(0, 0, w, h);
    for (let y = 8; y < h - 8; y += 16) {
      for (let x = 8; x < w - 8; x += 14) {
        const r = rnd();
        if (r < 0.42) {
          g.fillStyle = r < 0.12 ? "#ffd9a0" : r < 0.3 ? "#ffc271" : "#a9d3ff";
          g.globalAlpha = 0.55 + rnd() * 0.45;
          g.fillRect(x, y, 8, 9);
        }
      }
    }
    g.globalAlpha = 1;
  });
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function glowTexture() {
  return canvasTexture(64, 64, (g) => {
    const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    r.addColorStop(0, "rgba(255,255,255,1)");
    r.addColorStop(0.25, "rgba(255,255,255,.55)");
    r.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = r; g.fillRect(0, 0, 64, 64);
  });
}

// ---------------------------------------------------------------- scene
export class IntersectionScene {
  constructor(canvas, container) {
    this.canvas = canvas;
    this.container = container;
    this.cars = { ns: [], ew: [] };      // queued cars, front first
    this.moving = new Set();             // cars with an active tween (incl. leaving)
    this.onFrame = null;
    this.onDrop = null;
    this.clock = new THREE.Clock();

    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.outputColorSpace = THREE.SRGBColorSpace;

    const scene = (this.scene = new THREE.Scene());
    scene.background = skyTexture();
    scene.fog = new THREE.Fog("#1b2742", 110, 300);

    const cam = (this.camera = new THREE.PerspectiveCamera(40, 1, 0.5, 900));
    cam.position.set(-68, 58, 78);
    const controls = (this.controls = new OrbitControls(cam, canvas));
    controls.target.set(-15, 0, 13);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 30;
    controls.maxDistance = 190;
    controls.maxPolarAngle = Math.PI * 0.46;
    controls.enablePan = true;
    controls.update();

    this._lights();
    this._ground();
    this._markings();
    this._signals();
    this._scenery();

    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(scene, cam));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.75, 0.55, 0.82);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.setLights(0);
    r.setAnimationLoop(() => this._frame());
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.setSize(w, h);
    this.camera.aspect = w / h;
    // pull back a little on narrow screens so both queues stay in view
    this.camera.fov = w / h < 1 ? 52 : 40;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------ world
  _lights() {
    const s = this.scene;
    s.add(new THREE.HemisphereLight("#a9c6ff", "#2a2233", 1.25));
    const sun = new THREE.DirectionalLight("#ffc896", 2.4);
    sun.position.set(-70, 60, -40);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const c = sun.shadow.camera;
    c.left = -90; c.right = 90; c.top = 90; c.bottom = -90; c.near = 10; c.far = 260;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.4;
    s.add(sun);
    const fill = new THREE.DirectionalLight("#6f8fd6", 0.6);
    fill.position.set(60, 40, 80);
    s.add(fill);
  }

  _ground() {
    const s = this.scene;
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), new THREE.MeshStandardMaterial({ color: "#1f2a26", roughness: 1 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.05; ground.receiveShadow = true;
    s.add(ground);

    const tex = asphaltTexture();
    const mk = (w, d, rx, ry) => {
      const t = tex.clone(); t.needsUpdate = true; t.repeat.set(rx, ry);
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshStandardMaterial({ map: t, roughness: 0.92, metalness: 0.02 }));
      m.rotation.x = -Math.PI / 2; m.receiveShadow = true; return m;
    };
    const ns = mk(ROAD_HALF * 2, 600, 2, 75); ns.position.y = 0.0; s.add(ns);
    const ew = mk(600, ROAD_HALF * 2, 75, 2); ew.position.y = 0.005; s.add(ew);

    // sidewalks + grass plots in the four quadrants
    const walk = new THREE.MeshStandardMaterial({ color: "#4a5160", roughness: 0.95 });
    const curb = new THREE.MeshStandardMaterial({ color: "#6b7280", roughness: 0.8 });
    const grass = new THREE.MeshStandardMaterial({ color: "#23402f", roughness: 1 });
    const plaza = new THREE.MeshStandardMaterial({ color: "#3a404c", roughness: 0.95 });
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      const size = 300, off = ROAD_HALF + size / 2;
      const w = new THREE.Mesh(new THREE.BoxGeometry(size, 0.3, size), walk);
      w.position.set(sx * off, 0.15, sz * off); w.receiveShadow = true; s.add(w);
      const cb = new THREE.Mesh(new THREE.BoxGeometry(size, 0.34, size), curb);
      cb.position.set(sx * (off - 0.25), 0.12, sz * (off - 0.25)); s.add(cb);
      const park = sx < 0 && sz > 0;   // the south-west block (towards the camera) is a low park
      const inner = new THREE.Mesh(new THREE.BoxGeometry(size, 0.34, size), park ? grass : plaza);
      inner.position.set(sx * (off + 4), 0.17, sz * (off + 4)); inner.receiveShadow = true; s.add(inner);
    }
    // intersection box slightly darker
    const box = new THREE.Mesh(new THREE.PlaneGeometry(ROAD_HALF * 2, ROAD_HALF * 2), new THREE.MeshStandardMaterial({ color: "#2a2e35", roughness: 0.95 }));
    box.rotation.x = -Math.PI / 2; box.position.y = 0.01; box.receiveShadow = true;
    s.add(box);
  }

  _markings() {
    const s = this.scene;
    const white = new THREE.MeshStandardMaterial({ color: "#e5e7eb", roughness: 0.7 });
    const yellow = new THREE.MeshStandardMaterial({ color: "#f2b632", roughness: 0.7 });
    const flat = (w, d, mat, x, z, y = 0.02) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat);
      m.rotation.x = -Math.PI / 2; m.position.set(x, y, z); m.receiveShadow = true; s.add(m); return m;
    };
    // centre lines (double yellow) outside the box
    for (const sgn of [-1, 1]) {
      const len = 300 - STOP, c = sgn * (STOP + len / 2);
      flat(0.18, len, yellow, -0.2, c); flat(0.18, len, yellow, 0.2, c);
      flat(len, 0.18, yellow, c, -0.2); flat(len, 0.18, yellow, c, 0.2);
    }
    // lane edge dashes
    for (let d = STOP + 4; d < 300; d += 9) for (const sgn of [-1, 1]) {
      flat(0.16, 3.5, white, sgn * (ROAD_HALF - 0.6), d); flat(0.16, 3.5, white, sgn * (ROAD_HALF - 0.6), -d);
      flat(3.5, 0.16, white, d, sgn * (ROAD_HALF - 0.6)); flat(3.5, 0.16, white, -d, sgn * (ROAD_HALF - 0.6));
    }
    // zebra crossings
    for (let k = -ROAD_HALF + 1; k <= ROAD_HALF - 1; k += 1.6) {
      flat(0.8, 3, white, k, ROAD_HALF + 2.3); flat(0.8, 3, white, k, -(ROAD_HALF + 2.3));
      flat(3, 0.8, white, ROAD_HALF + 2.3, k); flat(3, 0.8, white, -(ROAD_HALF + 2.3), k);
    }
    // stop lines of the two controlled approaches glow with the signal colour
    this.stopMat = {
      ns: new THREE.MeshStandardMaterial({ color: "#ffffff", emissive: "#000000", roughness: 0.6 }),
      ew: new THREE.MeshStandardMaterial({ color: "#ffffff", emissive: "#000000", roughness: 0.6 }),
    };
    flat(ROAD_HALF - 0.4, 0.55, this.stopMat.ns, LANE, STOP - 0.2, 0.025);
    flat(0.55, ROAD_HALF - 0.4, this.stopMat.ew, -(STOP - 0.2), LANE, 0.025);
    flat(ROAD_HALF - 0.4, 0.55, white, -LANE, -(STOP - 0.2));
    flat(0.55, ROAD_HALF - 0.4, white, STOP - 0.2, -LANE);
    // arrows painted in the queue lanes
    const arrowShape = new THREE.Shape([[-0.35, 0], [0.35, 0], [0.35, 2], [0.9, 2], [0, 3.4], [-0.9, 2], [-0.35, 2]].map(([x, y]) => new THREE.Vector2(x, y)));
    const arrowGeo = new THREE.ShapeGeometry(arrowShape);
    const arrow = (x, z, yaw) => {
      const m = new THREE.Mesh(arrowGeo, white);
      m.rotation.set(-Math.PI / 2, 0, yaw); m.position.set(x, 0.022, z); s.add(m);
    };
    arrow(LANE, STOP + 38, 0);
    arrow(-(STOP + 38), LANE, -Math.PI / 2);
  }

  _signalHead(pos, facingYaw, withArm) {
    const s = this.scene;
    const g = new THREE.Group();
    const metal = new THREE.MeshStandardMaterial({ color: "#2b3038", roughness: 0.5, metalness: 0.6 });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.22, 7.2, 12), metal);
    pole.position.y = 3.6; pole.castShadow = true; g.add(pole);
    const head = new THREE.Group();
    head.position.set(0, 6.2, 0);
    head.scale.setScalar(1.35);
    if (withArm) {
      const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 5, 8), metal);
      arm.rotation.z = Math.PI / 2; arm.position.set(-2.5, 7, 0); g.add(arm);
      head.position.set(-4.6, 6.4, 0);
    }
    const housing = new THREE.Mesh(new THREE.BoxGeometry(0.85, 2.3, 0.7), new THREE.MeshStandardMaterial({ color: "#14171c", roughness: 0.6 }));
    housing.castShadow = true; head.add(housing);
    const board = new THREE.Mesh(new THREE.BoxGeometry(1.3, 2.75, 0.08), new THREE.MeshStandardMaterial({ color: "#0c0e11", roughness: 0.8 }));
    board.position.z = -0.4; head.add(board);
    const lamps = {};
    const glowTex = this._glowTex || (this._glowTex = glowTexture());
    [["red", "#ff3b30", 0.72], ["amber", "#ffb020", 0], ["green", "#22e06b", -0.72]].forEach(([name, col, y]) => {
      const mat = new THREE.MeshStandardMaterial({ color: "#1a1a1a", emissive: col, emissiveIntensity: 0.04, roughness: 0.3 });
      const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.25, 20, 12), mat);
      lamp.scale.z = 0.55; lamp.position.set(0, y, 0.33); head.add(lamp);
      const visor = new THREE.Mesh(new THREE.CylinderGeometry(0.33, 0.33, 0.35, 16, 1, true, 0, Math.PI), new THREE.MeshStandardMaterial({ color: "#0d0f12", side: THREE.DoubleSide }));
      visor.rotation.x = Math.PI / 2; visor.rotation.y = Math.PI / 2; visor.rotation.z = Math.PI / 2;
      visor.position.set(0, y + 0.04, 0.5); head.add(visor);
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: col, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
      glow.scale.set(2.2, 2.2, 1); glow.position.set(0, y, 0.6); head.add(glow);
      lamps[name] = { mat, glow };
    });
    g.add(head);
    g.position.copy(pos);
    g.rotation.y = facingYaw;
    s.add(g);
    return lamps;
  }

  _signals() {
    // yaw so the lamps (local +z) face the approaching traffic
    this.heads = {
      ns: [this._signalHead(new THREE.Vector3(ROAD_HALF + 1.6, 0.3, STOP + 0.8), 0, false),
           this._signalHead(new THREE.Vector3(-(ROAD_HALF + 1.6), 0.3, -(STOP + 0.8)), Math.PI, false),
           this._signalHead(new THREE.Vector3(-(ROAD_HALF + 1.6), 0.3, ROAD_HALF + 1.6), 0, true)],
      ew: [this._signalHead(new THREE.Vector3(-(STOP + 0.8), 0.3, ROAD_HALF + 1.6), -Math.PI / 2, false),
           this._signalHead(new THREE.Vector3(STOP + 0.8, 0.3, -(ROAD_HALF + 1.6)), Math.PI / 2, false),
           this._signalHead(new THREE.Vector3(ROAD_HALF + 1.6, 0.3, ROAD_HALF + 1.6), -Math.PI / 2, true)],
    };
  }

  _tree(x, z, h = 1) {
    const g = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.35, 2.2 * h, 7), this._trunkMat || (this._trunkMat = new THREE.MeshStandardMaterial({ color: "#4a3527", roughness: 1 })));
    trunk.position.y = 1.1 * h; trunk.castShadow = true; g.add(trunk);
    const leafMat = new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(0.3 + Math.random() * 0.06, 0.45, 0.2 + Math.random() * 0.08), roughness: 0.9, flatShading: true });
    const crown = new THREE.Mesh(new THREE.IcosahedronGeometry(1.9 * h, 0), leafMat);
    crown.position.y = 3.6 * h; crown.castShadow = true; g.add(crown);
    const crown2 = new THREE.Mesh(new THREE.IcosahedronGeometry(1.3 * h, 0), leafMat);
    crown2.position.set(0.6, 4.8 * h, 0.3); crown2.castShadow = true; g.add(crown2);
    g.position.set(x, 0.34, z); g.rotation.y = Math.random() * Math.PI;
    this.scene.add(g);
  }

  _streetLamp(x, z, yaw) {
    const metal = this._lampMetal || (this._lampMetal = new THREE.MeshStandardMaterial({ color: "#3a404a", metalness: 0.6, roughness: 0.45 }));
    const g = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.16, 8, 8), metal);
    pole.position.y = 4; g.add(pole);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.12, 0.12), metal);
    arm.position.set(1.1, 7.9, 0); g.add(arm);
    const bulb = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.18, 0.45), new THREE.MeshStandardMaterial({ color: "#fff", emissive: "#ffd9a0", emissiveIntensity: 2.2 }));
    bulb.position.set(2.1, 7.78, 0); g.add(bulb);
    g.position.set(x, 0.3, z); g.rotation.y = yaw;
    this.scene.add(g);
  }

  _scenery() {
    const s = this.scene;
    // buildings in the three far quadrants
    const rand = (a, b) => a + Math.random() * (b - a);
    const quads = [[1, -1], [-1, -1], [1, 1]];
    const palette = ["#2c3444", "#343c4d", "#283040", "#3a3f4a", "#2f3a4a"];
    let seed = 11;
    for (const [sx, sz] of quads) {
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        if (Math.random() < 0.18) continue;
        const w = rand(9, 14), d = rand(9, 14);
        const h = sx > 0 && sz > 0 ? rand(8, 20) : rand(10, 38) * (1 - 0.12 * Math.min(i, j));
        const x = sx * (ROAD_HALF + 12 + i * 17 + rand(-1, 1));
        const z = sz * (ROAD_HALF + 12 + j * 17 + rand(-1, 1));
        const tex = windowTexture(seed++); tex.repeat.set(Math.max(1, Math.round(w / 6)), Math.max(1, Math.round(h / 12)));
        const mat = new THREE.MeshStandardMaterial({ color: palette[(i + j) % palette.length], roughness: 0.85, metalness: 0.1, emissive: "#ffffff", emissiveMap: tex, emissiveIntensity: 0.9 });
        const roof = new THREE.MeshStandardMaterial({ color: "#1f2530", roughness: 0.9 });
        const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [mat, mat, roof, roof, mat, mat]);
        b.position.set(x, h / 2 + 0.34, z); b.castShadow = true; b.receiveShadow = true;
        s.add(b);
      }
    }
    // park in the camera-side quadrant
    for (let i = 0; i < 26; i++) {
      const x = -rand(ROAD_HALF + 14, 95), z = rand(ROAD_HALF + 14, 95);
      this._tree(x, z, rand(0.8, 1.25));
    }
    // street trees along the far pavements
    for (let d = 26; d < 120; d += 14) {
      this._tree(ROAD_HALF + 3.2, -d, 0.85); this._tree(d, -(ROAD_HALF + 3.2), 0.85);
    }
    for (let d = 22; d < 140; d += 24) {
      this._streetLamp(-(ROAD_HALF + 2), d, 0); this._streetLamp(ROAD_HALF + 2, -d, Math.PI);
      this._streetLamp(-d, -(ROAD_HALF + 2), -Math.PI / 2); this._streetLamp(d, ROAD_HALF + 2, Math.PI / 2);
    }
  }

  // ------------------------------------------------------------ cars
  _car() {
    const g = new THREE.Group();
    const kind = Math.random();
    const L = kind < 0.15 ? 5.0 : 4.3, W = 2.0, tall = kind < 0.15 ? 1.25 : kind < 0.4 ? 1.05 : 0.85;
    const color = kind > 0.93 ? "#f5c518" : CAR_COLORS[(Math.random() * CAR_COLORS.length) | 0];
    const paint = new THREE.MeshPhysicalMaterial({ color, roughness: 0.32, metalness: 0.55, clearcoat: 1, clearcoatRoughness: 0.15 });
    const glass = this._glass || (this._glass = new THREE.MeshPhysicalMaterial({ color: "#0d1520", roughness: 0.08, metalness: 0.9 }));
    const rubber = this._rubber || (this._rubber = new THREE.MeshStandardMaterial({ color: "#111", roughness: 0.9 }));
    const rim = this._rim || (this._rim = new THREE.MeshStandardMaterial({ color: "#9aa3ad", metalness: 0.9, roughness: 0.3 }));
    const head = this._headMat || (this._headMat = new THREE.MeshStandardMaterial({ color: "#fff", emissive: "#fff4dc", emissiveIntensity: 3 }));
    const tail = this._tailMat || (this._tailMat = new THREE.MeshStandardMaterial({ color: "#400", emissive: "#ff1a1a", emissiveIntensity: 2.2 }));

    const body = new THREE.Mesh(new THREE.BoxGeometry(W, tall * 0.8, L), paint);
    body.position.y = 0.55 + tall * 0.4; body.castShadow = true; g.add(body);
    const skirt = new THREE.Mesh(new THREE.BoxGeometry(W * 0.96, 0.25, L * 0.98), rubber);
    skirt.position.y = 0.5; g.add(skirt);
    const cabinL = kind < 0.15 ? L * 0.62 : L * 0.5;
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(W * 0.86, 0.72, cabinL), glass);
    cabin.position.set(0, 0.55 + tall * 0.8 + 0.34, kind < 0.15 ? 0.3 : 0.25); cabin.castShadow = true; g.add(cabin);
    const roof = new THREE.Mesh(new THREE.BoxGeometry(W * 0.8, 0.1, cabinL * 0.9), paint);
    roof.position.set(0, cabin.position.y + 0.4, cabin.position.z); g.add(roof);
    if (color === "#f5c518") {
      const sign = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.3, 0.3), new THREE.MeshStandardMaterial({ color: "#fff", emissive: "#ffe9a8", emissiveIntensity: 1.2 }));
      sign.position.set(0, roof.position.y + 0.2, roof.position.z); g.add(sign);
    }
    const wheelGeo = this._wheelGeo || (this._wheelGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.34, 18));
    const rimGeo = this._rimGeo || (this._rimGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.36, 10));
    g.userData.wheels = [];
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      const wg = new THREE.Group();
      const w = new THREE.Mesh(wheelGeo, rubber); w.rotation.z = Math.PI / 2; wg.add(w);
      const r = new THREE.Mesh(rimGeo, rim); r.rotation.z = Math.PI / 2; wg.add(r);
      wg.position.set(sx * (W / 2 - 0.08), 0.42, sz * (L / 2 - 0.85));
      g.add(wg); g.userData.wheels.push(wg);
    }
    for (const sx of [-1, 1]) {
      const h = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.16, 0.06), head);
      h.position.set(sx * 0.66, body.position.y + 0.08, -L / 2 - 0.02); g.add(h);
      const t = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.14, 0.06), tail);
      t.position.set(sx * 0.66, body.position.y + 0.12, L / 2 + 0.02); g.add(t);
    }
    g.userData.paint = paint;
    this.scene.add(g);
    return g;
  }

  _place(car, road, pos) {
    car.position.copy(pos);
    car.rotation.y = ROUTES[road].yaw;
  }

  _tween(car, to, delay, dur, easing, onDone) {
    const now = performance.now();
    car.userData.motion = { from: car.position.clone(), to: to.clone(), t0: now + delay, dur, ease: easing, onDone };
    this.moving.add(car);
  }

  _remove(car) {
    this.moving.delete(car);
    this.scene.remove(car);
    car.traverse((o) => {
      if (o.isMesh) {
        o.geometry !== this._wheelGeo && o.geometry !== this._rimGeo && o.geometry.dispose();
        if (o.userData.fadeMat) o.material.dispose();
      }
    });
    car.userData.paint?.dispose();
  }

  /** Put exactly ns / ew cars in the queues (no animation). */
  setQueues(ns, ew) {
    for (const road of ["ns", "ew"]) {
      this.cars[road].forEach((c) => this._remove(c));
      this.cars[road] = [];
    }
    for (const c of [...this.moving]) this._remove(c);
    for (let i = 0; i < ns; i++) { const c = this._car(); this._place(c, "ns", nsSlot(i)); this.cars.ns.push(c); }
    for (let i = 0; i < ew; i++) { const c = this._car(); this._place(c, "ew", ewSlot(i)); this.cars.ew.push(c); }
  }

  /** Lights: action 0 = EW green, 1 = NS green. amberFor = road currently changing to red. */
  setLights(action, amberFor = null) {
    const state = { ns: action === 1 ? "green" : "red", ew: action === 0 ? "green" : "red" };
    if (amberFor) state[amberFor] = "amber";
    for (const road of ["ns", "ew"]) {
      for (const lamps of this.heads[road]) {
        for (const [name, { mat, glow }] of Object.entries(lamps)) {
          const on = name === state[road];
          mat.emissiveIntensity = on ? 4.5 : 0.04;
          glow.material.opacity = on ? 0.9 : 0;
        }
      }
      const sm = this.stopMat[road];
      sm.emissive.set(state[road] === "green" ? "#22e06b" : state[road] === "amber" ? "#ffb020" : "#ff3b30");
      sm.emissiveIntensity = 0.55;
    }
  }

  /**
   * Animate one environment tick for one road.
   *  departFromQueue: cars leaving from the front of the queue
   *  passThrough:     new arrivals that drive straight through (queue was short)
   *  join:            new arrivals that join the back of the queue
   *  dropped:         new arrivals turned away because the queue is full
   */
  animateRoad(road, { departFromQueue, passThrough, join, dropped }, D) {
    const R = ROUTES[road];
    const q = this.cars[road];
    const leaving = q.splice(0, departFromQueue);
    leaving.forEach((car, i) => {
      this._tween(car, R.exit, i * D * 0.12, D * 0.85, ease.inQuad, () => this._remove(car));
    });
    // remaining cars roll forward
    q.forEach((car, i) => {
      const to = R.slot(i);
      if (car.position.distanceTo(to) > 0.01) this._tween(car, to, D * 0.18, D * 0.55, ease.inOutCubic);
    });
    for (let k = 0; k < passThrough; k++) {
      const car = this._car(); this._place(car, road, R.spawn.clone());
      this._tween(car, R.exit, k * D * 0.1, D * 0.95, ease.linear, () => this._remove(car));
    }
    for (let k = 0; k < join; k++) {
      const car = this._car(); this._place(car, road, R.spawn.clone().addScaledVector(R.back, k * 8));
      const idx = q.length;
      q.push(car);
      this._tween(car, R.slot(idx), D * 0.05 + k * D * 0.08, D * 0.8, ease.outCubic);
    }
    for (let k = 0; k < dropped; k++) {
      const car = this._car(); this._place(car, road, R.spawn.clone().addScaledVector(R.back, k * 8));
      const target = R.slot(q.length + 0.35 + k);
      this._tween(car, target, D * 0.1 + k * D * 0.1, D * 0.65, ease.outCubic, () => this._turnAway(car, D));
    }
  }

  _turnAway(car, D) {
    // the queue is full: the car flashes red and leaves the model
    this.onDrop?.(car.position.clone().setY(3));
    car.traverse((o) => {
      if (o.isMesh) {
        o.material = o.material.clone(); o.userData.fadeMat = true;
        o.material.transparent = true;
        if (o.material.emissive) { o.material.emissive.set("#ff2020"); o.material.emissiveIntensity = Math.max(o.material.emissiveIntensity, 0.6); }
      }
    });
    const t0 = performance.now(), dur = Math.max(300, D * 0.35);
    car.userData.fade = { t0, dur };
    this.moving.add(car);
  }

  // ------------------------------------------------------------ loop
  _frame() {
    const now = performance.now();
    for (const car of [...this.moving]) {
      const m = car.userData.motion;
      if (m) {
        const t = Math.min(1, Math.max(0, (now - m.t0) / m.dur));
        const prev = car.position.clone();
        car.position.lerpVectors(m.from, m.to, m.ease(t));
        const dist = car.position.distanceTo(prev);
        car.userData.wheels.forEach((w) => (w.rotation.x -= dist / 0.42));
        if (t >= 1) {
          car.userData.motion = null;
          if (!car.userData.fade) this.moving.delete(car);
          m.onDone?.();
        }
      }
      const f = car.userData.fade;
      if (f) {
        const t = Math.min(1, (now - f.t0) / f.dur);
        car.traverse((o) => { if (o.isMesh) o.material.opacity = 1 - t; });
        car.position.y = -t * 0.6;
        if (t >= 1) this._remove(car);
      }
    }
    this.controls.update();
    this.composer.render();
    this.onFrame?.();
  }

  /** Screen position (px, relative to the canvas) of a world point. */
  project(v) {
    const p = v.clone().project(this.camera);
    return { x: (p.x * 0.5 + 0.5) * this.container.clientWidth, y: (-p.y * 0.5 + 0.5) * this.container.clientHeight, visible: p.z < 1 };
  }

  queueAnchor(road, cap) {
    const p = ROUTES[road].slot(cap + 0.15);
    return p.setY(4.2);
  }
}
