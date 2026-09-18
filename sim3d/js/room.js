import * as THREE from "three";

export function buildDesk({ woodColor = 0x4a4a4a } = {}) {
  const desk = new THREE.Group();
  desk.name = "Desk";
  const wood = new THREE.MeshStandardMaterial({ color: woodColor, roughness: 0.55 });
  const top = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.06, 0.8), wood);
  top.position.y = 0.74;
  top.castShadow = true;
  top.receiveShadow = true;
  desk.add(top);
  for (const x of [-0.82, 0.82]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.71, 0.07), wood);
    leg.position.set(x, 0.355, 0.32);
    leg.castShadow = true;
    desk.add(leg);
    const leg2 = leg.clone();
    leg2.position.z = -0.32;
    desk.add(leg2);
  }

  const lamp = new THREE.Group();
  const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.02, 0.42, 8), new THREE.MeshStandardMaterial({ color: 0x222 }));
  arm.position.set(0.78, 1.02, 0.02);
  arm.rotation.z = 0.5;
  lamp.add(arm);
  const shade = new THREE.Mesh(
    new THREE.ConeGeometry(0.12, 0.14, 16, 1, true),
    new THREE.MeshStandardMaterial({ color: 0xd8d8d8, emissive: 0x444444, emissiveIntensity: 0.35, side: THREE.DoubleSide })
  );
  shade.position.set(0.68, 1.18, 0.02);
  shade.rotation.x = Math.PI;
  lamp.add(shade);
  desk.add(lamp);
  return desk;
}

export function buildRoom(scene) {
  const room = new THREE.Group();
  room.name = "Room";

  const wallMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.9 });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 6), new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.88 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  room.add(floor);

  const back = new THREE.Mesh(new THREE.PlaneGeometry(8, 3.2), wallMat);
  back.position.set(0, 1.6, -3);
  back.receiveShadow = true;
  room.add(back);
  const left = new THREE.Mesh(new THREE.PlaneGeometry(6, 3.2), wallMat);
  left.rotation.y = Math.PI / 2;
  left.position.set(-4, 1.6, 0);
  room.add(left);
  const right = left.clone();
  right.position.x = 4;
  right.rotation.y = -Math.PI / 2;
  room.add(right);

  const window = new THREE.Mesh(
    new THREE.PlaneGeometry(1.6, 1.1),
    new THREE.MeshStandardMaterial({ color: 0xc8c8c8, emissive: 0x222222, emissiveIntensity: 0.35, roughness: 0.3 })
  );
  window.position.set(-3.98, 1.8, -0.4);
  window.rotation.y = Math.PI / 2;
  room.add(window);

  const desk = buildDesk();
  desk.position.set(0, 0, -1.15);
  room.add(desk);

  const wood = new THREE.MeshStandardMaterial({ color: 0x4a4a4a, roughness: 0.55 });
  const chair = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.08, 0.42), wood);
  chair.position.set(0, 0.46, -0.35);
  chair.castShadow = true;
  room.add(chair);
  const backrest = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.46, 0.06), wood);
  backrest.position.set(0, 0.72, -0.14);
  room.add(backrest);

  scene.add(room);
  return { room, desk };
}
