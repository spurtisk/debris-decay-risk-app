/* Space debris decay risk: hero scene, lab, and the upload-driven pipeline views.
   Depends on Three.js (global THREE), pipeline.js (DebrisPipeline) and model.js (DebrisModel). */
(function () {
  "use strict";

  var R_EARTH = 6371.0;          // km
  var MU = 398600.4418;          // km^3 / s^2
  var ANALYSIS_MS = Date.UTC(2026, 8, 26);
  var DAY_MS = 86400000;
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY"];
  var ORBITS = ["GEO", "HEO", "LEO", "MEO"];
  var COUNTRIES = ["CHINA", "EU", "INDIA", "JAPAN", "OTHER", "RUSSIA", "USA"];

  function $(id) { return document.getElementById(id); }
  function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function easeInOut(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function fmt(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    return Number(v).toLocaleString("en-US", { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 });
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // Compressed altitude scale for the hero (sqrt keeps LEO, MEO and GEO distinct).
  function shellRadius(altKm) { return 1 + 0.6 * Math.sqrt(altKm / R_EARTH); }

  // ---------- shared Earth and atmosphere ----------
  var EARTH_VERT = [
    "varying vec3 vN; varying vec3 vP;",
    "void main(){ vN = normal; vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }"
  ].join("\n");

  var EARTH_FRAG = [
    "precision highp float;",
    "varying vec3 vN; varying vec3 vP;",
    "uniform vec3 uSun; uniform float uTime;",
    "float hash(vec3 p){ p = fract(p*0.3183099 + vec3(0.1)); p *= 17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }",
    "float noise(vec3 x){",
    "  vec3 i = floor(x); vec3 f = fract(x); f = f*f*(3.0-2.0*f);",
    "  float n000=hash(i), n100=hash(i+vec3(1,0,0)), n010=hash(i+vec3(0,1,0)), n110=hash(i+vec3(1,1,0));",
    "  float n001=hash(i+vec3(0,0,1)), n101=hash(i+vec3(1,0,1)), n011=hash(i+vec3(0,1,1)), n111=hash(i+vec3(1,1,1));",
    "  return mix(mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y), mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y), f.z);",
    "}",
    "float fbm(vec3 p){ float v=0.0; float a=0.5; for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.03; a*=0.5; } return v; }",
    "void main(){",
    "  vec3 p = normalize(vP);",
    "  float h = fbm(p*2.4);",
    "  float land = smoothstep(0.50, 0.58, h);",
    "  vec3 ocean = mix(vec3(0.02,0.11,0.24), vec3(0.04,0.22,0.40), fbm(p*7.0));",
    "  vec3 ground = mix(vec3(0.14,0.30,0.20), vec3(0.42,0.35,0.24), smoothstep(0.6,0.74,h));",
    "  vec3 base = mix(ocean, ground, land);",
    "  base = mix(base, vec3(0.88,0.94,1.0), smoothstep(0.80,0.94,abs(p.y))*0.9);",
    "  float cl = smoothstep(0.56, 0.82, fbm(p*3.6 + vec3(uTime*0.015, 0.0, uTime*0.01)));",
    "  float diff = max(dot(normalize(vN), normalize(uSun)), 0.0);",
    "  float light = 0.05 + 0.95*diff;",
    "  vec3 col = base*light;",
    "  col = mix(col, vec3(0.95)*(0.12+0.88*diff), cl*0.7);",
    "  gl_FragColor = vec4(col, 1.0);",
    "}"
  ].join("\n");

  var ATMO_VERT = [
    "varying vec3 vN;",
    "void main(){ vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }"
  ].join("\n");

  var ATMO_FRAG = [
    "varying vec3 vN;",
    "void main(){ float i = pow(0.7 - dot(vN, vec3(0.0,0.0,1.0)), 2.4); gl_FragColor = vec4(0.42,0.80,1.0,1.0) * i * 1.15; }"
  ].join("\n");

  function makeEarth(scene) {
    var uniforms = { uSun: { value: new THREE.Vector3(0.9, 0.35, 0.6).normalize() }, uTime: { value: 0 } };
    var earth = new THREE.Mesh(
      new THREE.SphereGeometry(1, 96, 64),
      new THREE.ShaderMaterial({ vertexShader: EARTH_VERT, fragmentShader: EARTH_FRAG, uniforms: uniforms })
    );
    scene.add(earth);
    var atmo = new THREE.Mesh(
      new THREE.SphereGeometry(1.085, 64, 48),
      new THREE.ShaderMaterial({
        vertexShader: ATMO_VERT, fragmentShader: ATMO_FRAG, side: THREE.BackSide,
        transparent: true, blending: THREE.AdditiveBlending, depthWrite: false
      })
    );
    scene.add(atmo);
    return { uniforms: uniforms };
  }

  function makeStars(scene, count, rMin, rMax) {
    var pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
    for (var i = 0; i < count; i++) {
      var u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      var r = lerp(rMin, rMax, Math.random());
      pos[i * 3] = r * s * Math.cos(t); pos[i * 3 + 1] = r * u; pos[i * 3 + 2] = r * s * Math.sin(t);
      var warm = Math.random() < 0.15;
      var b = 0.55 + Math.random() * 0.45;
      col[i * 3] = b * (warm ? 1.0 : 0.85); col[i * 3 + 1] = b * (warm ? 0.9 : 0.92); col[i * 3 + 2] = b;
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    var m = new THREE.PointsMaterial({ size: 0.9, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false });
    var pts = new THREE.Points(g, m);
    scene.add(pts);
    return pts;
  }

  function ringGeometry(radius, segments) {
    var pts = [];
    for (var i = 0; i <= segments; i++) {
      var a = (i / segments) * Math.PI * 2;
      pts.push(new THREE.Vector3(radius * Math.cos(a), radius * Math.sin(a), 0));
    }
    return new THREE.BufferGeometry().setFromPoints(pts);
  }

  function tiltX(v, incl) {
    var c = Math.cos(incl), s = Math.sin(incl);
    return new THREE.Vector3(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
  }

  function sphericalPosition(dist, az, el) {
    return new THREE.Vector3(dist * Math.cos(el) * Math.cos(az), dist * Math.sin(el), dist * Math.cos(el) * Math.sin(az));
  }

  function fitRenderer(renderer, camera, canvas) {
    var w = canvas.clientWidth || 300, h = canvas.clientHeight || 300;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  // ==================================================================
  // HERO: zoomable Earth, orbit shells and drifting debris
  // ==================================================================
  var hero = {};

  function initHero() {
    var canvas = $("heroCanvas");
    if (!canvas || typeof THREE === "undefined") return;
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(38, 1, 0.1, 500);
    var earth = makeEarth(scene);
    makeStars(scene, 2600, 60, 140);

    var shells = [
      { alt: 550, incl: 53, color: 0x7fd6ff, opacity: 0.32 },
      { alt: 20200, incl: 55, color: 0x7fd6ff, opacity: 0.22 },
      { alt: 35786, incl: 0, color: 0x6fe3c1, opacity: 0.30 }
    ];
    shells.forEach(function (s) {
      var geo = ringGeometry(shellRadius(s.alt), 240);
      var mat = new THREE.LineBasicMaterial({ color: s.color, transparent: true, opacity: s.opacity });
      var line = new THREE.LineLoop(geo, mat);
      line.rotation.x = s.incl * Math.PI / 180;
      scene.add(line);
    });

    var N = 2600;
    var orbits = new Array(N), posArr = new Float32Array(N * 3), colArr = new Float32Array(N * 3);
    var inclChoices = [53, 66, 79, 87, 98];
    for (var i = 0; i < N; i++) {
      var alt = lerp(380, 1150, Math.pow(Math.random(), 1.4));
      var incl = inclChoices[Math.floor(Math.random() * inclChoices.length)] + (Math.random() - 0.5) * 4;
      var r = shellRadius(alt);
      orbits[i] = {
        r: r,
        incl: incl * Math.PI / 180,
        raan: Math.random() * Math.PI * 2,
        theta: Math.random() * Math.PI * 2,
        speed: 0.35 / Math.pow(r, 1.5) * (0.8 + Math.random() * 0.4)
      };
      var b = 0.5 + Math.random() * 0.5;
      colArr[i * 3] = b; colArr[i * 3 + 1] = b * 0.62; colArr[i * 3 + 2] = b * 0.36;
    }
    var debrisGeo = new THREE.BufferGeometry();
    var debrisPos = new THREE.BufferAttribute(posArr, 3);
    debrisPos.setUsage(THREE.DynamicDrawUsage);
    debrisGeo.setAttribute("position", debrisPos);
    debrisGeo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    var debris = new THREE.Points(debrisGeo, new THREE.PointsMaterial({
      size: 0.018, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false
    }));
    scene.add(debris);

    var ember = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({
      color: 0xff6b4a, size: 0.05, sizeAttenuation: true, transparent: true, opacity: 0.95, depthWrite: false
    }));
    var emberPos = new Float32Array(3 * 3);
    ember.geometry.setAttribute("position", new THREE.BufferAttribute(emberPos, 3));
    scene.add(ember);
    var emberIdx = [0, 1, 2];
    orbits[0].r = shellRadius(560); orbits[0].incl = 53 * Math.PI / 180; orbits[0].speed = 0.12;
    orbits[1].r = shellRadius(700); orbits[1].incl = 98 * Math.PI / 180; orbits[1].speed = 0.1;
    orbits[2].r = shellRadius(880); orbits[2].incl = 66 * Math.PI / 180; orbits[2].speed = 0.11;

    var geoSat = new THREE.Mesh(new THREE.SphereGeometry(0.022, 16, 12), new THREE.MeshBasicMaterial({ color: 0xe8f7ff }));
    scene.add(geoSat);
    var geoR = shellRadius(35786);

    var stage = canvas.parentElement;
    var labelEmber = document.createElement("div"); labelEmber.className = "label-tag"; labelEmber.textContent = "High-risk fragment (example)";
    var labelGeo = document.createElement("div"); labelGeo.className = "label-tag"; labelGeo.textContent = "GEO payload";
    var labelLeo = document.createElement("div"); labelLeo.className = "label-tag"; labelLeo.textContent = "LEO shell, 550 km";
    stage.appendChild(labelEmber); stage.appendChild(labelGeo); stage.appendChild(labelLeo);

    var state = { dragX: 0, dragY: 0, targetX: 0, targetY: 0, dragging: false, lastX: 0, lastY: 0 };
    canvas.addEventListener("pointerdown", function (e) {
      state.dragging = true; state.lastX = e.clientX; state.lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", function (e) {
      if (!state.dragging) return;
      state.targetX += (e.clientX - state.lastX) * 0.005;
      state.targetY = clamp(state.targetY + (e.clientY - state.lastY) * 0.003, -0.5, 0.6);
      state.lastX = e.clientX; state.lastY = e.clientY;
    });
    canvas.addEventListener("pointerup", function () { state.dragging = false; });
    canvas.addEventListener("pointercancel", function () { state.dragging = false; });

    hero = { renderer: renderer, scene: scene, camera: camera, earth: earth, canvas: canvas,
      orbits: orbits, debris: debris, debrisPos: posArr, ember: ember, emberPos: emberPos, emberIdx: emberIdx,
      geoSat: geoSat, geoR: geoR, labels: { ember: labelEmber, geo: labelGeo, leo: labelLeo }, state: state,
      section: $("hero"), progress: $("heroProgress"), copy: $("heroCopy"), visible: true };
    fitRenderer(renderer, camera, canvas);
    window.addEventListener("resize", function () { fitRenderer(renderer, camera, canvas); });
  }

  function updateHero(dt, now) {
    if (!hero.renderer || !hero.visible) return;
    var sec = hero.section;
    var total = sec.offsetHeight - window.innerHeight;
    var p = clamp(-sec.getBoundingClientRect().top / Math.max(total, 1), 0, 1);
    var eased = easeInOut(p);
    var st = hero.state;
    st.dragX = lerp(st.dragX, st.targetX, 0.08);
    st.dragY = lerp(st.dragY, st.targetY, 0.08);
    var az = 0.6 + st.dragX + (reduceMotion ? 0 : now * 0.00004);
    var el = 0.32 + st.dragY;
    var dist = lerp(11.5, 2.6, eased);
    hero.camera.position.copy(sphericalPosition(dist, az, el));
    hero.camera.lookAt(0, 0, 0);

    var r = hero.orbits;
    var pos = hero.debrisPos;
    for (var i = 0; i < r.length; i++) {
      var o = r[i];
      if (!reduceMotion) o.theta += o.speed * dt;
      var x = o.r * Math.cos(o.theta), y = o.r * Math.sin(o.theta);
      var yc = y * Math.cos(o.incl), zc = y * Math.sin(o.incl);
      var cO = Math.cos(o.raan), sO = Math.sin(o.raan);
      pos[i * 3] = x * cO - yc * sO;
      pos[i * 3 + 1] = x * sO + yc * cO;
      pos[i * 3 + 2] = zc;
    }
    hero.debris.geometry.attributes.position.needsUpdate = true;

    var ep = hero.emberPos;
    for (var k = 0; k < 3; k++) {
      var oo = r[hero.emberIdx[k]];
      var xx = oo.r * Math.cos(oo.theta), yy = oo.r * Math.sin(oo.theta);
      var yy2 = yy * Math.cos(oo.incl), zz2 = yy * Math.sin(oo.incl);
      var cc = Math.cos(oo.raan), ss = Math.sin(oo.raan);
      ep[k * 3] = xx * cc - yy2 * ss; ep[k * 3 + 1] = xx * ss + yy2 * cc; ep[k * 3 + 2] = zz2;
    }
    hero.ember.geometry.attributes.position.needsUpdate = true;

    var ga = now * 0.00012 * (reduceMotion ? 0 : 1);
    hero.geoSat.position.set(hero.geoR * Math.cos(ga), 0, hero.geoR * Math.sin(ga));

    hero.earth.uniforms.uTime.value = now * 0.001;
    hero.copy.style.opacity = String(1 - clamp((p - 0.05) / 0.3, 0, 1));
    hero.progress.style.width = (p * 100).toFixed(2) + "%";

    hero.renderer.render(hero.scene, hero.camera);

    placeLabel(hero.labels.ember, hero.ember.geometry.attributes.position, 0, hero.camera, hero.canvas);
    placeLabel(hero.labels.geo, null, null, hero.camera, hero.canvas, hero.geoSat.position);
    placeLabel(hero.labels.leo, null, null, hero.camera, hero.canvas, new THREE.Vector3(shellRadius(550), 0, 0));
  }

  function placeLabel(el, attr, index, camera, canvas, vec) {
    var v = vec;
    if (!v) { v = new THREE.Vector3(attr.getX(index), attr.getY(index), attr.getZ(index)); }
    var p = v.clone().project(camera);
    var w = canvas.clientWidth, h = canvas.clientHeight;
    var x = (p.x * 0.5 + 0.5) * w, y = (-p.y * 0.5 + 0.5) * h;
    var visible = p.z < 1 && x > 0 && x < w && y > 0 && y < h;
    el.style.display = visible ? "block" : "none";
    el.style.left = x.toFixed(1) + "px";
    el.style.top = (y - 14).toFixed(1) + "px";
  }

  // ==================================================================
  // LAB: orbit viewer and model readout
  // ==================================================================
  var lab = {};

  function initLab() {
    var canvas = $("labCanvas");
    if (!canvas || typeof THREE === "undefined") return;
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    var earth = makeEarth(scene);
    makeStars(scene, 900, 30, 60);

    var orbitGeo = new THREE.BufferGeometry();
    var orbitMat = new THREE.LineBasicMaterial({ color: 0x7fd6ff, transparent: true, opacity: 0.9 });
    var orbitLine = new THREE.LineLoop(orbitGeo, orbitMat);
    scene.add(orbitLine);

    var markerMat = new THREE.MeshBasicMaterial({ color: 0x6fe3c1 });
    var marker = new THREE.Mesh(new THREE.SphereGeometry(0.035, 24, 16), markerMat);
    scene.add(marker);
    var glow = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 16), new THREE.MeshBasicMaterial({
      color: 0x6fe3c1, transparent: true, opacity: 0.22, depthWrite: false, blending: THREE.AdditiveBlending
    }));
    scene.add(glow);

    var perigeeMarker = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 10), new THREE.MeshBasicMaterial({ color: 0xffb36b }));
    var apogeeMarker = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 10), new THREE.MeshBasicMaterial({ color: 0xffb36b, transparent: true, opacity: 0.6 }));
    scene.add(perigeeMarker); scene.add(apogeeMarker);

    var state = { az: 0.7, el: 0.35, dist: 5.2, dragging: false, lastX: 0, lastY: 0 };
    canvas.addEventListener("pointerdown", function (e) {
      state.dragging = true; state.lastX = e.clientX; state.lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", function (e) {
      if (!state.dragging) return;
      state.az += (e.clientX - state.lastX) * 0.008;
      state.el = clamp(state.el + (e.clientY - state.lastY) * 0.006, -1.2, 1.2);
      state.lastX = e.clientX; state.lastY = e.clientY;
    });
    canvas.addEventListener("pointerup", function () { state.dragging = false; });
    canvas.addEventListener("pointercancel", function () { state.dragging = false; });
    canvas.addEventListener("wheel", function (e) {
      e.preventDefault();
      state.dist = clamp(state.dist + e.deltaY * 0.004, 3.0, 12);
    }, { passive: false });

    lab = { renderer: renderer, scene: scene, camera: camera, earth: earth, canvas: canvas,
      orbitGeo: orbitGeo, orbitLine: orbitLine, marker: marker, glow: glow, markerMat: markerMat,
      perigeeMarker: perigeeMarker, apogeeMarker: apogeeMarker, state: state,
      segments: 320, visible: false, scale: 1, incl: 0, rp: 1, ra: 1, e: 0, p: 1, riskValue: 0 };
    fitRenderer(renderer, camera, canvas);
    window.addEventListener("resize", function () { fitRenderer(renderer, camera, canvas); });
  }

  // Physics readouts from the inputs (no model needed)
  function physics(perigeeKm, ecc, launchStr) {
    var rp = R_EARTH + perigeeKm;
    var ra = rp * (1 + ecc) / (1 - ecc);
    var a = (rp + ra) / 2;
    var periodMin = 2 * Math.PI * Math.sqrt(a * a * a / MU) / 60;
    var vP = Math.sqrt(MU * (2 / rp - 1 / a));
    var parts = launchStr.split("-").map(Number);
    var launchMs = Date.UTC(parts[0], parts[1] - 1, parts[2]);
    var ageYears = Math.floor((ANALYSIS_MS - launchMs) / DAY_MS) / 365.25;
    return { rp: rp, ra: ra, a: a, apogeeAlt: ra - R_EARTH, periodMin: periodMin, vP: vP, ageYears: ageYears };
  }

  function updateLabOrbit(ph, inclDeg) {
    if (!lab.renderer) return;
    var e = ph.ra === ph.rp ? 0 : (ph.ra - ph.rp) / (ph.ra + ph.rp);
    var p = ph.a * (1 - e * e);
    var k = 0.9 / Math.max(ph.apogeeAlt, 300);
    var incl = inclDeg * Math.PI / 180;
    var N = lab.segments;
    var arr = new Float32Array((N + 1) * 3);
    var rsMap = function (rKm) { return 1 + (rKm - R_EARTH) * k; };
    for (var i = 0; i <= N; i++) {
      var th = (i / N) * Math.PI * 2;
      var r = p / (1 + e * Math.cos(th));
      var rs = rsMap(r);
      var v = tiltX(new THREE.Vector3(rs * Math.cos(th), rs * Math.sin(th), 0), incl);
      arr[i * 3] = v.x; arr[i * 3 + 1] = v.y; arr[i * 3 + 2] = v.z;
    }
    lab.orbitGeo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    lab.orbitGeo.attributes.position.needsUpdate = true;
    lab.scale = k; lab.incl = incl; lab.e = e; lab.p = p;
    lab.perigeeMarker.position.copy(tiltX(new THREE.Vector3(rsMap(ph.rp), 0, 0), incl));
    lab.apogeeMarker.position.copy(tiltX(new THREE.Vector3(-rsMap(ph.ra), 0, 0), incl));
  }

  function updateLab(now) {
    if (!lab.renderer || !lab.visible) return;
    var st = lab.state;
    var az = st.az + (reduceMotion ? 0 : now * 0.00003);
    lab.camera.position.copy(sphericalPosition(st.dist, az, st.el));
    lab.camera.lookAt(0, 0, 0);
    var th = (now * 0.00012 * (reduceMotion ? 0 : 1)) % (Math.PI * 2);
    var e = lab.e || 0, p = lab.p || 1;
    var r = p / (1 + e * Math.cos(th));
    var rs = 1 + (r - R_EARTH) * lab.scale;
    var v = tiltX(new THREE.Vector3(rs * Math.cos(th), rs * Math.sin(th), 0), lab.incl);
    lab.marker.position.copy(v);
    lab.glow.position.copy(v);
    var color = new THREE.Color().setHSL(lerp(0.45, 0.04, clamp(lab.riskValue || 0, 0, 1)), 0.85, 0.6);
    lab.markerMat.color.copy(color);
    lab.glow.material.color.copy(color);
    lab.earth.uniforms.uTime.value = now * 0.001;
    lab.renderer.render(lab.scene, lab.camera);
  }

  // ==================================================================
  // Results from the uploaded file
  // ==================================================================
  var models = null;      // prepared model object for DebrisModel
  var results = null;     // full pipeline result

  var FEATURE_LABELS = {
    perigee_km: "Perigee", eccentricity: "Eccentricity", inclination_deg: "Inclination",
    log_rcs: "Log RCS", object_age_years: "Object age"
  };
  function featureLabel(name) {
    if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
    var eq = name.indexOf("=");
    if (eq > 0) {
      var col = name.slice(0, eq), lvl = name.slice(eq + 1);
      var prefix = { type: "Type", orbit_class: "Orbit", source_country: "Country" }[col] || col;
      return prefix + ": " + lvl;
    }
    return name;
  }

  function readInputs() {
    return {
      type: $("sType").value,
      orbit_class: $("sOrbit").value,
      source_country: $("sCountry").value,
      perigee_km: Number($("rPerigee").value),
      eccentricity: Number($("rEcc").value),
      inclination_deg: Number($("rIncl").value),
      rcs_m2: Math.pow(10, Number($("rRcs").value)),
      launch_date: $("dLaunch").value
    };
  }

  function setOutputs(out) {
    var clsLr = $("outLrClass"), clsRf = $("outRfClass");
    if (!out) {
      $("outRisk").textContent = "–"; $("outLr").textContent = "–"; $("outRf").textContent = "–";
      clsLr.textContent = "Upload the dataset to train"; clsRf.textContent = "Upload the dataset to train";
      $("riskFill").style.width = "0%";
      $("contrib").innerHTML = '<p class="dim" style="font-size:0.9rem">The breakdown appears once the models are trained.</p>';
      return;
    }
    $("outRisk").textContent = out.predicted_decay_risk.toFixed(4);
    $("riskFill").style.width = (clamp(out.predicted_decay_risk, 0, 1) * 100).toFixed(1) + "%";
    $("outLr").textContent = (out.p_high_logistic * 100).toFixed(1) + "%";
    $("outRf").textContent = (out.p_high_forest * 100).toFixed(1) + "%";
    clsLr.innerHTML = '<span class="pill ' + (out.class_logistic === "high" ? "high" : "normal") + '">' + (out.class_logistic === "high" ? "high risk" : "normal") + "</span> at threshold " + models.logistic.threshold.toFixed(3);
    clsRf.innerHTML = '<span class="pill ' + (out.class_forest === "high" ? "high" : "normal") + '">' + (out.class_forest === "high" ? "high risk" : "normal") + "</span> at threshold " + models.forest.threshold.toFixed(3);

    var contrib = out.logistic_contributions.slice().sort(function (a, b) { return Math.abs(b.contribution) - Math.abs(a.contribution); }).slice(0, 7);
    var maxAbs = Math.max.apply(null, contrib.map(function (c) { return Math.abs(c.contribution); }).concat([1e-9]));
    $("contrib").innerHTML = contrib.map(function (c) {
      var pct = Math.min(50, Math.abs(c.contribution) / maxAbs * 50);
      var positive = c.contribution >= 0;
      var left = positive ? 50 : 50 - pct;
      var color = positive ? "var(--ember)" : "var(--teal)";
      return '<div class="crow"><span class="name" title="' + esc(featureLabel(c.feature)) + '">' + esc(featureLabel(c.feature)) +
        '</span><span class="track"><span class="fill" style="left:' + left.toFixed(2) + '%;width:' + pct.toFixed(2) + '%;background:' + color + '"></span></span>' +
        '<span class="num">' + (c.contribution >= 0 ? "+" : "") + c.contribution.toFixed(2) + "</span></div>";
    }).join("");
  }

  function drawSensitivity(input) {
    var svg = $("spark");
    if (!models) { svg.innerHTML = ""; return; }
    var W = 520, H = 180, padL = 40, padB = 26, padT = 10, padR = 12;
    var lo = -3, hi = 1.5, N = 60;
    var lr = [], rf = [];
    for (var i = 0; i <= N; i++) {
      var lv = lerp(lo, hi, i / N);
      var o = window.DebrisModel.predict(models, Object.assign({}, input, { rcs_m2: Math.pow(10, lv) }));
      lr.push([lv, o.p_high_logistic]); rf.push([lv, o.p_high_forest]);
    }
    function X(v) { return padL + (v - lo) / (hi - lo) * (W - padL - padR); }
    function Y(v) { return padT + (1 - v) * (H - padT - padB); }
    function path(arr) { return arr.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }).join(" "); }
    var grid = [0, 0.25, 0.5, 0.75, 1].map(function (g) {
      return '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + Y(g) + '" y2="' + Y(g) + '" stroke="rgba(160,200,255,0.1)"/>' +
        '<text x="' + (padL - 8) + '" y="' + (Y(g) + 4) + '" fill="#8d9bb3" font-size="11" text-anchor="end" font-family="IBM Plex Mono">' + g.toFixed(2) + "</text>";
    }).join("");
    var xt = [-3, -2, -1, 0, 1].map(function (t) {
      return '<text x="' + X(t) + '" y="' + (H - 6) + '" fill="#8d9bb3" font-size="11" text-anchor="middle" font-family="IBM Plex Mono">' + Math.pow(10, t).toString() + "</text>";
    }).join("");
    var cur = Math.log10(input.rcs_m2);
    svg.innerHTML = grid + xt +
      '<path d="' + path(lr) + '" fill="none" stroke="#6fe3c1" stroke-width="2.2"/>' +
      '<path d="' + path(rf) + '" fill="none" stroke="#ffb36b" stroke-width="2.2" stroke-dasharray="5 4"/>' +
      '<line x1="' + X(clamp(cur, lo, hi)) + '" x2="' + X(clamp(cur, lo, hi)) + '" y1="' + padT + '" y2="' + (H - padB) + '" stroke="#7fd6ff" stroke-opacity="0.8"/>' +
      '<text x="' + (W - padR) + '" y="' + (padT + 12) + '" fill="#8d9bb3" font-size="11" text-anchor="end" font-family="IBM Plex Mono">RCS (m²), log scale</text>';
  }

  function update() {
    var input = readInputs();
    $("vPerigee").textContent = input.perigee_km.toFixed(0) + " km";
    $("vEcc").textContent = input.eccentricity.toFixed(3);
    $("vIncl").textContent = input.inclination_deg.toFixed(1) + "°";
    $("vRcs").textContent = input.rcs_m2.toFixed(3) + " m²";

    var ph = physics(input.perigee_km, input.eccentricity, input.launch_date);
    $("pApogee").textContent = Math.round(ph.apogeeAlt).toLocaleString("en-US") + " km";
    $("pPeriod").textContent = ph.periodMin.toFixed(1) + " min";
    $("pSpeed").textContent = ph.vP.toFixed(2) + " km/s";
    $("pAge").textContent = ph.ageYears.toFixed(1) + " yr";
    updateLabOrbit(ph, input.inclination_deg);

    var out = null;
    if (models) {
      out = window.DebrisModel.predict(models, input);
      lab.riskValue = out.predicted_decay_risk;
    }
    setOutputs(out);
    drawSensitivity(input);
  }

  var PRESETS = {
    payload: { type: "PAYLOAD", orbit_class: "LEO", source_country: "USA", perigee: 550, ecc: 0.001, incl: 53, rcs: -0.3, launch: "2020-01-01" },
    rocket: { type: "ROCKET BODY", orbit_class: "LEO", source_country: "RUSSIA", perigee: 320, ecc: 0.002, incl: 65, rcs: 0.48, launch: "1998-06-01" },
    fragment: { type: "DEBRIS", orbit_class: "LEO", source_country: "CHINA", perigee: 780, ecc: 0.01, incl: 98, rcs: -1.7, launch: "2009-02-10" },
    eccentric: { type: "DEBRIS", orbit_class: "HEO", source_country: "EU", perigee: 300, ecc: 0.25, incl: 63, rcs: -0.7, launch: "2015-07-01" }
  };

  function applyPreset(key) {
    var p = PRESETS[key];
    if (!p) return;
    $("sType").value = p.type; $("sOrbit").value = p.orbit_class; $("sCountry").value = p.source_country;
    $("rPerigee").value = p.perigee; $("rEcc").value = p.ecc; $("rIncl").value = p.incl;
    $("rRcs").value = p.rcs; $("dLaunch").value = p.launch;
    update();
  }

  function fillSelect(sel, values, selected) {
    sel.innerHTML = "";
    values.forEach(function (v) {
      var o = document.createElement("option");
      o.value = v; o.textContent = v;
      if (v === selected) o.selected = true;
      sel.appendChild(o);
    });
  }

  // ---------- pipeline walkthrough ----------
  var STAGE_TEXT = [
    "The uploaded CSV is read exactly as it is. Nothing is changed or dropped at this point, so the row count here is the count of the file.",
    "Bronze keeps every value as text. Blank or NA-like fields are marked missing, and nothing has been cleaned yet.",
    "Silver converts types, parses dates and applies rules C0 to C14. Rejected rows are counted by the rule that removed them. Outliers are flagged and kept, because a large object is a real risk.",
    "Gold builds summary tables for reporting and one feature table. The feature table holds the model inputs, the labels and the split, so every model reads the same rows.",
    "Three models are trained and then scored on the same held-out objects, so their results can be compared directly."
  ];
  var STAGE_TITLES = ["Raw CSV", "01 Bronze", "02 Silver", "03 Gold", "04 Models"];

  function stageKV(i) {
    if (!results) return [["Waiting for a file", "–"]];
    var R = results;
    if (i === 0) return [["Rows", fmt(R.rawRows)], ["Columns", fmt(R.rawColumns)], ["File", R.fileName || "–"]];
    if (i === 1) return [["Rows", fmt(R.rawRows)], ["Columns, read as text", fmt(R.rawColumns)], ["Blank or NA fields", fmt(R.rawMissing)]];
    if (i === 2) {
      var outliers = R.silver.filter(function (r) { return r.rcs_outlier; }).length;
      return [["Kept", fmt(R.silver.length)], ["Rejected", fmt(R.rejectCount)], ["IQR outliers flagged", fmt(outliers)], ["High-risk objects", fmt(R.split.trainHigh + R.split.testHigh)]];
    }
    if (i === 3) return [["Summary rows", fmt(R.gold.summary.length)], ["Country rows", fmt(R.gold.country.length)], ["Feature rows", fmt(R.silver.length)], ["Train share", "0.80"]];
    return [["Models", "3"], ["Training rows", fmt(R.split.train)], ["Test rows", fmt(R.split.test)]];
  }

  function renderStage(i) {
    var btns = document.querySelectorAll(".stage");
    btns.forEach(function (b, j) { b.setAttribute("aria-pressed", j === i ? "true" : "false"); });
    $("stageCard").innerHTML = '<span class="eyebrow">Stage ' + (i + 1) + " of 5</span>" +
      '<h3 style="margin-top:10px">' + STAGE_TITLES[i] + "</h3>" +
      '<p class="muted" style="margin-top:10px">' + STAGE_TEXT[i] + "</p>" +
      '<div class="kvs" style="margin-top:14px">' + stageKV(i).map(function (kv) {
        return '<div class="kv"><span>' + esc(kv[0]) + "</span><span>" + esc(kv[1]) + "</span></div>";
      }).join("") + "</div>";
  }

  // Rules C0 to C14 with a plain-English line for each.
  var AUDIT = [
    ["C0", "Read the CSV", "Every line must parse to the same number of fields as the header."],
    ["C2", "Remove exact duplicates", "Drops rows that repeat every column value."],
    ["C3", "Unique object ID", "Keeps the first row for each object_id."],
    ["C4", "Clean category text", "Trims spaces and uppercases type, orbit and country."],
    ["C5", "Merge synonyms and rare countries", "Rocket-body spellings become ROCKET BODY. Countries under 1 percent of rows become OTHER."],
    ["C6", "Convert numeric columns", "Values that cannot be read as numbers become empty."],
    ["C7", "Parse dates", "Parses launch_date and checks the launch year against it."],
    ["C8", "Known orbit and type", "Rows with an unknown orbit class or object type are removed."],
    ["C9", "Fill or reject missing values", "Missing RCS is filled with the median for its object type. Missing decay risk is rejected."],
    ["C10", "Impossible values", "Eight range checks, such as perigee above 100 km and a launch date on or before 26 Sep 2026."],
    ["C11", "Kepler period check", "The stored period must match the semi-major axis within 1 percent."],
    ["C12", "Flag RCS outliers", "Flags rows outside the IQR fences and keeps them. Adds log₁₀ of RCS."],
    ["C13", "Derived features", "Altitude band, log decay risk, and the high-risk flag for the top 10 percent."],
    ["C14", "Final Silver schema", "Keeps the columns the rest of the pipeline needs."]
  ];

  function renderAudit() {
    var rowsByCode = {}, notesByCode = {};
    if (results) {
      results.audit.forEach(function (a) {
        var code = a.step.split(" ")[0];
        rowsByCode[code] = (rowsByCode[code] || 0) + a.rows_removed;
        notesByCode[code] = a.note;
      });
    }
    $("audit").innerHTML = AUDIT.map(function (a) {
      var removed = rowsByCode[a[0]] || 0;
      var label = results ? (removed ? "−" + fmt(removed) + " rows" : "0 removed") : "–";
      return '<button class="audit-row' + (removed ? " hit" : "") + '" style="text-align:left;font:inherit;color:inherit;cursor:pointer;width:100%" aria-expanded="false" data-why="' + esc(a[2]) + (results && notesByCode[a[0]] ? " " + esc(notesByCode[a[0]]) : "") + '">' +
        '<span class="code">' + a[0] + "</span><span>" + a[1] + '</span><span class="rm">' + label + "</span></button>";
    }).join("");
    document.querySelectorAll(".audit-row").forEach(function (row) {
      row.addEventListener("click", function () {
        var open = row.getAttribute("aria-expanded") === "true";
        document.querySelectorAll(".audit-row").forEach(function (r) {
          r.setAttribute("aria-expanded", "false");
          var n = r.querySelector(".why"); if (n) n.remove();
        });
        if (!open) {
          row.setAttribute("aria-expanded", "true");
          var note = document.createElement("span");
          note.className = "why";
          note.textContent = row.getAttribute("data-why");
          row.appendChild(note);
        }
      });
    });
  }

  function renderTable(el, headers, numericCols, rows) {
    var head = "<thead><tr>" + headers.map(function (h, j) {
      return '<th' + (numericCols.indexOf(j) >= 0 ? ' class="n"' : "") + ">" + esc(h) + "</th>";
    }).join("") + "</tr></thead>";
    var body = "<tbody>" + rows.map(function (row) {
      return "<tr>" + row.map(function (cell, j) {
        return '<td' + (numericCols.indexOf(j) >= 0 ? ' class="n"' : "") + ">" + esc(cell) + "</td>";
      }).join("") + "</tr>";
    }).join("") + "</tbody>";
    el.innerHTML = head + body;
  }

  function renderGoldAndSilver(R) {
    renderTable($("goldSummary"),
      ["Altitude band", "Orbit", "Type", "Objects", "Median decay risk", "High-risk share", "Mean log RCS"], [3, 4, 5, 6],
      R.gold.summary.map(function (g) {
        return [g.altitude_band, g.orbit_class, g.type, fmt(g.objects), g.median_decay_risk.toExponential(2),
          (g.high_risk_share * 100).toFixed(1) + "%", g.mean_log_rcs.toFixed(3)];
      }));
    renderTable($("goldCountry"),
      ["Source country", "Objects", "High-risk objects", "Median decay risk"], [1, 2, 3],
      R.gold.country.map(function (g) {
        return [g.source_country, fmt(g.objects), fmt(g.high_risk_objects), g.median_decay_risk.toExponential(2)];
      }));
    renderTable($("silverPreview"),
      ["Object", "Type", "Orbit", "Country", "Perigee km", "Inclination", "Eccentricity", "Log RCS", "Decay risk", "High risk", "Band"],
      [4, 5, 6, 7, 8, 9],
      R.silver.slice(0, 8).map(function (r) {
        return [r.object_id, r.type, r.orbit_class, r.source_country, fmt(r.perigee_km, 1), fmt(r.inclination_deg, 2),
          r.eccentricity.toFixed(4), r.log_rcs === null ? "–" : r.log_rcs.toFixed(3), r.decay_risk.toExponential(2),
          String(r.high_decay_risk), r.altitude_band];
      }));
    renderTable($("rejectTable"), ["Step", "Reason", "Rows"], [2],
      R.rejectSummary.map(function (x) { return [x.step, x.reason, fmt(x.count)]; }).concat(
        R.rejectSummary.length ? [] : [["–", "no rows rejected", "0"]]));
    $("goldArea").hidden = false;
  }

  // ---------- model results ----------
  var SHORT = { "R2 log scale": 1, "R2 original scale": 1, "ROC-AUC": 1, "F1": 1, "PR-AUC": 1 };

  function renderModelResults(R) {
    var metric = {};
    R.models.metrics.forEach(function (m) { metric[m.model + "|" + m.metric] = m.value; });

    document.querySelectorAll("[data-metric]").forEach(function (el) {
      var v = metric[el.getAttribute("data-metric")];
      el.textContent = v === undefined ? "–" : v.toFixed(4);
    });
    document.querySelectorAll("[data-bar]").forEach(function (el) {
      var v = metric[el.getAttribute("data-bar")];
      el.style.width = v === undefined ? "0%" : (clamp(v, 0, 1) * 100).toFixed(2) + "%";
    });
    document.querySelectorAll("[data-cmp]").forEach(function (el) {
      var v = metric[el.getAttribute("data-cmp")];
      el.textContent = v === undefined ? "–" : v.toFixed(4);
    });
    document.querySelectorAll("[data-delta]").forEach(function (el) {
      var v = metric[el.getAttribute("data-delta")];
      var p2 = parseFloat(el.getAttribute("data-p2"));
      if (v === undefined) { el.textContent = "–"; return; }
      var d = v - p2;
      el.textContent = (d >= 0 ? "+" : "−") + Math.abs(d).toFixed(4);
    });

  }


  // ---------- added visuals: importance, calibration, precision-recall, heatmap ----------
  function drawImportance(R) {
    var rows = R.models.rf_importance.slice(0, 10);
    var max = Math.max.apply(null, rows.map(function (r) { return r.importance; }).concat([1e-9]));
    $("importanceBars").innerHTML = rows.map(function (r) {
      return '<div class="bar-line"><span>' + esc(featureLabel(r.feature)) + '</span><div class="bt"><div class="bf" style="width:' +
        (r.importance / max * 100).toFixed(1) + '%;background:var(--debris)"></div></div><span class="bv">' + r.importance.toFixed(3) + "</span></div>";
    }).join("");
  }

  function drawCalibration(R) {
    var svg = $("calSvg"), p = R.scores.lr, y = R.scores.actual;
    var W = 520, H = 300, pl = 44, pr = 14, pt = 14, pb = 40, bins = 10;
    var n = new Array(bins).fill(0), sp = new Array(bins).fill(0), sy = new Array(bins).fill(0);
    for (var i = 0; i < p.length; i++) {
      var b = Math.min(bins - 1, Math.floor(p[i] * bins));
      n[b]++; sp[b] += p[i]; sy[b] += y[i];
    }
    function X(v) { return pl + v * (W - pl - pr); }
    function Y(v) { return H - pb - v * (H - pt - pb); }
    var parts = [];
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) {
      parts.push('<line x1="' + X(g) + '" x2="' + X(g) + '" y1="' + pt + '" y2="' + (H - pb) + '" stroke="rgba(160,200,255,0.08)"/>');
      parts.push('<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + Y(g) + '" y2="' + Y(g) + '" stroke="rgba(160,200,255,0.08)"/>');
      parts.push('<text x="' + X(g) + '" y="' + (H - 22) + '" fill="#8d9bb3" font-size="11" text-anchor="middle" font-family="IBM Plex Mono">' + Math.round(g * 100) + "%</text>");
      parts.push('<text x="' + (pl - 6) + '" y="' + (Y(g) + 4) + '" fill="#8d9bb3" font-size="11" text-anchor="end" font-family="IBM Plex Mono">' + Math.round(g * 100) + "%</text>");
    });
    parts.push('<line x1="' + X(0) + '" y1="' + Y(0) + '" x2="' + X(1) + '" y2="' + Y(1) + '" stroke="#7fd6ff" stroke-dasharray="6 4" stroke-opacity="0.7"/>');
    parts.push('<text x="' + (pl + (W - pl - pr) / 2) + '" y="' + (H - 4) + '" fill="#8d9bb3" font-size="11" text-anchor="middle">predicted chance (average in group)</text>');
    parts.push('<text x="12" y="' + (pt + 110) + '" fill="#8d9bb3" font-size="11" text-anchor="middle" transform="rotate(-90 12 ' + (pt + 110) + ')">observed share high risk</text>');
    var maxN = Math.max.apply(null, n.concat([1]));
    var gap = 0, total = p.length;
    for (b = 0; b < bins; b++) {
      if (!n[b]) continue;
      var mp = sp[b] / n[b], mo = sy[b] / n[b];
      gap += n[b] / total * Math.abs(mo - mp);
      parts.push('<circle cx="' + X(mp).toFixed(1) + '" cy="' + Y(mo).toFixed(1) + '" r="' + (4 + 8 * Math.sqrt(n[b] / maxN)).toFixed(1) +
        '" fill="#6fe3c1" fill-opacity="0.75" stroke="#0b1020"/>');
    }
    svg.innerHTML = parts.join("");
    $("calGap").textContent = "Average gap between predicted and observed: " + (gap * 100).toFixed(1) +
      " percentage points. Smaller is more trustworthy. Circle size shows how many objects fall in each group.";
  }

  function prPoints(score, y) {
    var n = score.length, idx = [], i, P = 0;
    for (i = 0; i < n; i++) { idx.push(i); if (y[i]) P++; }
    idx.sort(function (a, b) { return score[b] - score[a]; });
    var tp = 0, fp = 0, pts = [];
    for (i = 0; i < n;) {
      var t = score[idx[i]];
      while (i < n && score[idx[i]] === t) { if (y[idx[i]]) tp++; else fp++; i++; }
      pts.push([tp / P, tp / (tp + fp)]);
    }
    return pts;
  }

  function drawPR(R) {
    var svg = $("prSvg"), y = R.scores.actual;
    var W = 520, H = 300, pl = 44, pr = 14, pt = 14, pb = 40;
    function X(v) { return pl + v * (W - pl - pr); }
    function Y(v) { return H - pb - v * (H - pt - pb); }
    var parts = [];
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) {
      parts.push('<line x1="' + X(g) + '" x2="' + X(g) + '" y1="' + pt + '" y2="' + (H - pb) + '" stroke="rgba(160,200,255,0.08)"/>');
      parts.push('<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + Y(g) + '" y2="' + Y(g) + '" stroke="rgba(160,200,255,0.08)"/>');
      parts.push('<text x="' + X(g) + '" y="' + (H - 22) + '" fill="#8d9bb3" font-size="11" text-anchor="middle" font-family="IBM Plex Mono">' + g.toFixed(2) + "</text>");
      parts.push('<text x="' + (pl - 6) + '" y="' + (Y(g) + 4) + '" fill="#8d9bb3" font-size="11" text-anchor="end" font-family="IBM Plex Mono">' + g.toFixed(2) + "</text>");
    });
    var base = y.reduce(function (s, v) { return s + v; }, 0) / y.length;
    parts.push('<line x1="' + X(0) + '" y1="' + Y(base) + '" x2="' + X(1) + '" y2="' + Y(base) + '" stroke="#8d9bb3" stroke-dasharray="5 4"/>');
    parts.push('<text x="' + (X(1) - 4) + '" y="' + (Y(base) - 6) + '" fill="#8d9bb3" font-size="11" text-anchor="end">coin toss</text>');
    [["lr", "#6fe3c1"], ["rf", "#ffb36b"]].forEach(function (m) {
      var pts = prPoints(R.scores[m[0]], y);
      var d = pts.map(function (q, k) { return (k ? "L" : "M") + X(q[0]).toFixed(1) + "," + Y(q[1]).toFixed(1); }).join(" ");
      parts.push('<path d="' + d + '" fill="none" stroke="' + m[1] + '" stroke-width="2.2"/>');
    });
    // marker: the slider threshold for the chosen model
    var which = plain.thrModel, thr = Number($("thrSlider").value), score = R.scores[which];
    var tp = 0, fp = 0, fn = 0;
    for (var i = 0; i < score.length; i++) {
      var pred = score[i] >= thr;
      if (y[i] && pred) tp++; else if (!y[i] && pred) fp++; else if (y[i]) fn++;
    }
    if (tp + fp > 0 && tp + fn > 0) {
      var rec = tp / (tp + fn), prec = tp / (tp + fp);
      parts.push('<circle cx="' + X(rec).toFixed(1) + '" cy="' + Y(prec).toFixed(1) + '" r="6" fill="none" stroke="#fff" stroke-width="2"/>');
    }
    parts.push('<text x="' + (pl + (W - pl - pr) / 2) + '" y="' + (H - 4) + '" fill="#8d9bb3" font-size="11" text-anchor="middle">recall: share of dangerous objects caught</text>');
    parts.push('<text x="12" y="' + (pt + 110) + '" fill="#8d9bb3" font-size="11" text-anchor="middle" transform="rotate(-90 12 ' + (pt + 110) + ')">precision</text>');
    svg.innerHTML = parts.join("");
  }

  // Altitude rows: 250 km steps up to 2,000 km, then one row for everything above.
  function altRow(base) {
    if (base >= 2000) return { key: 2000, label: "2,000 km and above" };
    var lo = Math.floor(base / 250) * 250;
    return { key: lo, label: fmt(lo) + "–" + fmt(lo + 250) + " km" };
  }

  function drawHeatmap(R) {
    var cells = {}, rows = {};
    R.gold.summary.forEach(function (g) {
      var base = parseInt(g.altitude_band, 10);
      var row = altRow(base);
      rows[row.key] = row.label;
      var k = row.key + "|" + g.orbit_class;
      var c = cells[k] || (cells[k] = { n: 0, h: 0 });
      c.n += g.objects; c.h += g.high_risk_share * g.objects;
    });
    var keys = Object.keys(rows).map(Number).sort(function (a, b) { return a - b; });
    var head = "<tr><th class=\"row\">Altitude</th>" + ORBITS.map(function (o) { return "<th>" + o + "</th>"; }).join("") + "</tr>";
    var body = keys.map(function (band) {
      return "<tr><th class=\"row\">" + esc(rows[band]) + "</th>" + ORBITS.map(function (o) {
        var c = cells[band + "|" + o];
        if (!c || !c.n) return '<td class="empty">–</td>';
        var share = c.h / c.n;
        return '<td style="background:rgba(255,107,74,' + (0.08 + share * 0.82).toFixed(3) + ')" title="' + fmt(c.n) + ' objects">' +
          (share * 100).toFixed(0) + "%</td>";
      }).join("") + "</tr>";
    }).join("");
    $("heatGrid").innerHTML = '<table class="heat">' + head + body + "</table>";
  }

  // ---------- plain-language results ----------
  var plain = { hist: "lr", thrModel: "lr", examples: [] };

  function pct(v) { return (v * 100).toFixed(1) + "%"; }

  function drawHistogram(R) {
    var svg = $("histSvg");
    var which = plain.hist;
    var score = R.scores[which], actual = R.scores.actual;
    var thr = R.thresholds[which];
    var W = 520, H = 260, pl = 44, pr = 14, pt = 14, pb = 40, bins = 20;
    var safe = new Array(bins).fill(0), high = new Array(bins).fill(0);
    var nSafe = 0, nHigh = 0;
    for (var i = 0; i < score.length; i++) {
      var b = Math.min(bins - 1, Math.floor(score[i] * bins));
      if (actual[i]) { high[b]++; nHigh++; } else { safe[b]++; nSafe++; }
    }
    var max = 1;
    for (var k = 0; k < bins; k++) max = Math.max(max, safe[k] + high[k]);
    var plotH = H - pt - pb, plotW = W - pl - pr, bw = plotW / bins;
    var parts = [];
    for (var j = 0; j < bins; j++) {
      var x = pl + j * bw + 1;
      var hS = safe[j] / max * plotH, hH = high[j] / max * plotH;
      if (hS > 0) parts.push('<rect x="' + x.toFixed(1) + '" y="' + (H - pb - hS).toFixed(1) + '" width="' + (bw - 2).toFixed(1) + '" height="' + hS.toFixed(1) + '" fill="#6fe3c1" fill-opacity="0.85"/>');
      if (hH > 0) parts.push('<rect x="' + x.toFixed(1) + '" y="' + (H - pb - hS - hH).toFixed(1) + '" width="' + (bw - 2).toFixed(1) + '" height="' + hH.toFixed(1) + '" fill="#ff6b4a" fill-opacity="0.9"/>');
    }
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      var xx = pl + t * plotW;
      parts.push('<text x="' + xx.toFixed(1) + '" y="' + (H - 16) + '" fill="#8d9bb3" font-size="11" text-anchor="middle" font-family="IBM Plex Mono">' + Math.round(t * 100) + "%</text>");
    });
    parts.push('<text x="' + (pl + plotW / 2) + '" y="' + (H - 2) + '" fill="#8d9bb3" font-size="11" text-anchor="middle">' +
      (which === "lr" ? "model's chance of high risk" : "share of 300 trees voting high") + "</text>");
    parts.push('<text x="' + (pl - 6) + '" y="' + (pt + 10) + '" fill="#8d9bb3" font-size="11" text-anchor="end" font-family="IBM Plex Mono">' + fmt(max) + "</text>");
    parts.push('<text x="' + (pl - 6) + '" y="' + (H - pb) + '" fill="#8d9bb3" font-size="11" text-anchor="end" font-family="IBM Plex Mono">0</text>');
    var tx = pl + thr * plotW;
    parts.push('<line x1="' + tx.toFixed(1) + '" x2="' + tx.toFixed(1) + '" y1="' + pt + '" y2="' + (H - pb) + '" stroke="#7fd6ff" stroke-width="2" stroke-dasharray="6 4"/>');
    parts.push('<text x="' + (tx + 6).toFixed(1) + '" y="' + (pt + 12) + '" fill="#7fd6ff" font-size="11" font-family="IBM Plex Mono">threshold ' + thr.toFixed(3) + "</text>");
    svg.innerHTML = parts.join("");
    $("histSafe").textContent = fmt(nSafe) + " test objects";
    $("histHigh").textContent = fmt(nHigh) + " test objects";
  }

  function updateThreshold(R) {
    var which = plain.thrModel;
    var score = R.scores[which], actual = R.scores.actual;
    var slider = $("thrSlider");
    var thr = Number(slider.value);
    $("thrVal").textContent = thr.toFixed(3);
    var tp = 0, fp = 0, fn = 0, tn = 0;
    for (var i = 0; i < score.length; i++) {
      var pred = score[i] >= thr;
      if (actual[i]) { if (pred) tp++; else fn++; }
      else { if (pred) fp++; else tn++; }
    }
    $("oTP").textContent = fmt(tp);
    $("oFN").textContent = fmt(fn);
    $("oFP").textContent = fmt(fp);
    $("oTN").textContent = fmt(tn);
    var name = which === "lr" ? "The logistic model" : "The Random Forest";
    var tuned = Math.abs(thr - R.thresholds[which]) < 0.0025
      ? " This line is where the model was tuned to sit, using training data only."
      : "";
    $("thrSentence").textContent = name + " warns about " + fmt(tp) + " of the " + fmt(tp + fn) +
      " dangerous test objects and misses " + fmt(fn) + ". It also raises " + fmt(fp) + " false alarms among the " +
      fmt(fp + tn) + " safe ones." + tuned;
    drawHistogram(R);
    drawPR(R);
  }

  function sizeWord(rcs) {
    if (rcs < 0.1) return "very small";
    if (rcs < 1) return "small";
    if (rcs < 10) return "medium";
    return "large";
  }

  function leafVote(tree, x) {
    var node = 0;
    while (tree.f[node] !== -2) node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    return tree.p[node];
  }

  function showExample(R, idx) {
    var row = R.testRows[idx];
    var input = {
      type: row.type, orbit_class: row.orbit_class, source_country: row.source_country,
      perigee_km: row.perigee_km, eccentricity: row.eccentricity, inclination_deg: row.inclination_deg,
      rcs_m2: row.rcs_m2, launch_date: row.launch_date
    };
    var out = window.DebrisModel.predict(models, input);
    var x = window.DebrisModel.buildRow(models.forest_row, input);
    var votes = models.forest.trees.map(function (t) { return leafVote(t, x) >= 0.5 ? 1 : 0; });
    var nHigh = votes.reduce(function (a, b) { return a + b; }, 0);
    var actualHigh = row.high_decay_risk === 1;
    var lrHigh = out.p_high_logistic >= models.logistic.threshold;
    var rfHigh = out.p_high_forest >= models.forest.threshold;

    $("exSteps").innerHTML = [
      "<li><b>The record.</b> A " + esc(row.type.toLowerCase()) + " in a " + esc(row.orbit_class) + " orbit, from " + esc(row.source_country) + ".</li>",
      "<li><b>Its path.</b> Lowest point " + fmt(row.perigee_km) + " km, eccentricity " + row.eccentricity.toFixed(3) +
        " (0 is a perfect circle), tilted " + row.inclination_deg.toFixed(1) + "° to the equator.</li>",
      "<li><b>Its size.</b> Radar cross-section " + row.rcs_m2.toFixed(3) + " m², which is " + sizeWord(row.rcs_m2) +
        " for this catalogue. Bigger objects reflect more radar and are easier to track.</li>",
      "<li><b>Its age.</b> About " + row.object_age_years.toFixed(1) + " years since launch, measured at 26 September 2026.</li>",
      "<li><b>The logistic model</b> gives it a " + pct(out.p_high_logistic) + " chance of high risk. That is " +
        (lrHigh ? "above" : "below") + " its threshold of " + models.logistic.threshold.toFixed(3) + ".</li>",
      "<li><b>The Random Forest</b> has " + nHigh + " of its 300 trees vote high risk, so its score is " + pct(out.p_high_forest) + ".</li>",
      "<li><b>The real label.</b> This object is " + (actualHigh ? "in" : "outside") + " the top ten percent by decay risk (decay risk " +
        row.decay_risk.toFixed(4) + ").</li>"
    ].join("");

    $("exVotes").innerHTML = votes.map(function (v) { return '<span class="' + (v ? "high" : "") + '"></span>'; }).join("");

    var agree = (lrHigh === actualHigh) && (rfHigh === actualHigh);
    $("exVerdict").textContent = (agree
      ? "Both models agree with the real label for this object."
      : "The models disagree with the real label for this object, or with each other. This is the kind of case the threshold slider trades off.") +
      " Logistic says " + (lrHigh ? "high risk" : "normal") + ", Random Forest says " + (rfHigh ? "high risk" : "normal") +
      ", and the label is " + (actualHigh ? "high risk" : "normal") + ".";
  }

  function renderPlain(R) {
    $("plainEmpty").hidden = true;
    drawHistogram(R);
    $("thrSlider").value = R.thresholds[plain.thrModel];
    updateThreshold(R);

    var highIdx = [], safeIdx = [];
    R.scores.actual.forEach(function (a, i) { (a ? highIdx : safeIdx).push(i); });
    function pick(list, n) {
      var out = [];
      for (var k = 0; k < n && list.length; k++) out.push(list[Math.floor(k * list.length / n)]);
      return out;
    }
    plain.examples = pick(highIdx, 4).concat(pick(safeIdx, 4));
    $("exSelect").innerHTML = plain.examples.map(function (idx) {
      var r = R.testRows[idx];
      var label = r.high_decay_risk ? "high risk" : "normal";
      return '<option value="' + idx + '">' + esc(r.object_id) + " · " + esc(r.type.toLowerCase()) + " · " + esc(r.orbit_class) + " · real label: " + label + "</option>";
    }).join("");
    if (plain.examples.length) showExample(R, plain.examples[0]);
  }

  function setupPlain() {
    document.querySelectorAll("[data-hmodel]").forEach(function (b) {
      b.addEventListener("click", function () {
        plain.hist = b.getAttribute("data-hmodel");
        document.querySelectorAll("[data-hmodel]").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        if (results) drawHistogram(results);
      });
    });
    document.querySelectorAll("[data-tmodel]").forEach(function (b) {
      b.addEventListener("click", function () {
        plain.thrModel = b.getAttribute("data-tmodel");
        document.querySelectorAll("[data-tmodel]").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        if (results) { $("thrSlider").value = results.thresholds[plain.thrModel]; updateThreshold(results); }
      });
    });
    $("thrSlider").addEventListener("input", function () { if (results) updateThreshold(results); });
    $("exSelect").addEventListener("change", function () { if (results && models) showExample(results, Number(this.value)); });
    $("exRandom").addEventListener("click", function () {
      if (!results || !models) return;
      var idx = Math.floor(Math.random() * results.testRows.length);
      var sel = $("exSelect");
      var opt = Array.prototype.find.call(sel.options, function (o) { return o.value === String(idx); });
      if (!opt) {
        opt = document.createElement("option");
        opt.value = String(idx);
        opt.textContent = results.testRows[idx].object_id + " · random test object";
        sel.appendChild(opt);
      }
      sel.value = String(idx);
      showExample(results, idx);
    });
  }

  // ---------- upload ----------
  function setStatus(message, isError) {
    var el = $("runStatus");
    el.textContent = message;
    el.classList.toggle("error", !!isError);
  }
  function setProgress(pct) { $("runBar").style.width = clamp(pct, 0, 100).toFixed(1) + "%"; }

  function setStagePlaceholders() {
    var subs = { 0: "–", 1: "–", 2: "–", 3: "–", 4: "–" };
    document.querySelectorAll(".stage").forEach(function (b) {
      var i = Number(b.getAttribute("data-stage"));
      b.querySelector(".sub").textContent = subs[i];
    });
  }

  function applyResults(R) {
    results = R;
    models = window.DebrisModel.prepare(R.models);
    $("modelBanner").hidden = true;
    $("runSummary").hidden = false;
    $("sRows").textContent = fmt(R.rawRows);
    $("sKept").textContent = fmt(R.silver.length);
    $("sRejected").textContent = fmt(R.rejectCount);
    $("sHigh").textContent = fmt(R.split.trainHigh + R.split.testHigh);
    $("sSplit").textContent = fmt(R.split.train) + " / " + fmt(R.split.test);
    $("sThrLR").textContent = R.thresholds.lr.toFixed(3);
    $("sThrRF").textContent = R.thresholds.rf.toFixed(3);
    $("sTime").textContent = R.seconds.toFixed(1) + " s";

    document.querySelectorAll(".stage").forEach(function (b) {
      var i = Number(b.getAttribute("data-stage"));
      var subs = [
        fmt(R.rawRows) + " rows", "every column as text", fmt(R.silver.length) + " rows kept",
        fmt(R.gold.summary.length) + " summary rows", "3 models, " + fmt(R.split.test) + " test rows"
      ];
      b.querySelector(".sub").textContent = subs[i];
    });

    safely("stage card", function () { renderStage(currentStage); });
    safely("audit", renderAudit);
    safely("gold and silver tables", function () { renderGoldAndSilver(R); });
    safely("model results", function () { renderModelResults(R); });
    safely("plain results", function () { renderPlain(R); });
    safely("importance chart", function () { drawImportance(R); });
    safely("calibration chart", function () { drawCalibration(R); });
    safely("precision-recall chart", function () { drawPR(R); });
    safely("heatmap", function () { drawHeatmap(R); });
    safely("lab outputs", update);
  }

  var currentStage = 0;
  var busy = false;

  async function handleFile(file) {
    if (!file || busy) return;
    busy = true;
    $("csvFile").disabled = true;
    setStatus("Reading " + file.name + " …", false);
    setProgress(2);
    try {
      var text = await file.text();
      var R = await window.DebrisPipeline.run(text, {
        fileName: file.name,
        onProgress: function (label, pct) { setStatus(label + " …", false); setProgress(pct); }
      });
      R.fileName = file.name;
      applyResults(R);
      setStatus("Finished in " + R.seconds.toFixed(1) + " s. The models below now use your file.", false);
      setProgress(100);
    } catch (err) {
      var isData = err && err.name === "DataError";
      setStatus(isData ? err.message : "The run stopped unexpectedly: " + (err && err.message ? err.message : err), true);
      setProgress(0);
      if (!isData) console.error(err);
    } finally {
      busy = false;
      $("csvFile").disabled = false;
      $("csvFile").value = "";
    }
  }

  function setupUpload() {
    var input = $("csvFile"), zone = $("dropZone");
    input.addEventListener("change", function () { handleFile(input.files && input.files[0]); });
    ["dragenter", "dragover"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove("over"); });
    });
    zone.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      handleFile(f);
    });
    zone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
  }

  // ==================================================================
  // Setup and loop
  // ==================================================================
  function setupControls() {
    fillSelect($("sType"), TYPES, "PAYLOAD");
    fillSelect($("sOrbit"), ORBITS, "LEO");
    fillSelect($("sCountry"), COUNTRIES, "USA");
    ["sType", "sOrbit", "sCountry", "rPerigee", "rEcc", "rIncl", "rRcs", "dLaunch"].forEach(function (id) {
      $(id).addEventListener("input", update);
      $(id).addEventListener("change", update);
    });
    document.querySelectorAll("[data-preset]").forEach(function (b) {
      b.addEventListener("click", function () { applyPreset(b.getAttribute("data-preset")); });
    });
    document.querySelectorAll(".stage").forEach(function (b, i) {
      b.addEventListener("click", function () { currentStage = i; renderStage(i); });
    });
    renderStage(0);
    renderAudit();
    setStagePlaceholders();
    setupUpload();
    setupPlain();
  }

  function setupObservers() {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) en.target.classList.add("in"); });
    }, { threshold: 0.12 });
    document.querySelectorAll(".reveal").forEach(function (el) { io.observe(el); });

    if (hero.canvas) {
      new IntersectionObserver(function (entries) { hero.visible = entries[0].isIntersecting; }).observe(hero.canvas);
    }
    if (lab.canvas) {
      new IntersectionObserver(function (entries) { lab.visible = entries[0].isIntersecting; }).observe(lab.canvas);
    }
  }

  var last = performance.now();
  function frame(now) {
    var dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    try { updateHero(dt, now); } catch (err) { /* keep the page alive */ }
    try { updateLab(now); } catch (err) { /* keep the page alive */ }
    requestAnimationFrame(frame);
  }

  // Each step is isolated, so a failure in the 3D scenes (for example, no WebGL in
  // the viewer) cannot stop the upload, the pipeline or the charts from working.
  function safely(name, fn) {
    try { fn(); } catch (err) { console.warn(name + " failed:", err); }
  }

  function boot() {
    safely("hero scene", initHero);
    safely("lab scene", initLab);
    safely("controls", setupControls);
    safely("observers", setupObservers);
    var banner = $("modelBanner");
    banner.hidden = false;
    banner.innerHTML = "<strong>Models not trained yet.</strong> Upload the Kaggle CSV in section 00 above. The pipeline runs in your browser, and the lab then scores objects with the three models. The orbit and physics readouts work without it.";
    safely("first update", update);
    requestAnimationFrame(frame);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
