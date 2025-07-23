/**
 * Helical Blockchain 3D Visualization
 * 
 * Creates an interactive 3D visualization of the helical blockchain
 * using Three.js, showing multiple strands, cross-links, and real-time updates.
 */

class HelixVisualization {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.container = null;
        
        // Helix parameters
        this.helixRadius = 10;
        this.helixPitch = 5;
        this.strandCount = 5;
        this.maxBlocks = 50;
        
        // Visualization objects
        this.strandGroups = {};
        this.blockMeshes = [];
        this.linkMeshes = [];
        this.animationId = null;
        this.autoRotateEnabled = false;
        
        // Colors for different strands
        this.strandColors = {
            coherence: 0x6366f1,   // Purple
            interaction: 0x06b6d4, // Cyan
            wisdom: 0x10b981,      // Green
            moral: 0xf59e0b,       // Orange
            social: 0x8b5cf6       // Violet
        };
        
        // Animation state
        this.rotationSpeed = 0.005;
        this.time = 0;
    }
    
    async init() {
        try {
            this.container = document.getElementById('helix-visualization');
            if (!this.container) {
                console.error('Helix visualization container not found');
                return;
            }
            
            this.setupThreeJS();
            this.createHelixStructure();
            this.setupLighting();
            this.setupControls();
            this.startAnimation();
            
            // Load initial blockchain data
            await this.loadBlockchainData();
            
            console.log('✅ Helix visualization initialized');
        } catch (error) {
            console.error('Failed to initialize helix visualization:', error);
        }
    }
    
    setupThreeJS() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0f0f23);
        
        // Camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(25, 15, 25);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
        
        // Handle resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    createHelixStructure() {
        // Create central axis
        this.createCentralAxis();
        
        // Create strand guides
        this.createStrandGuides();
        
        // Initialize strand groups
        Object.keys(this.strandColors).forEach(strandName => {
            this.strandGroups[strandName] = new THREE.Group();
            this.strandGroups[strandName].userData = { strand: strandName };
            this.scene.add(this.strandGroups[strandName]);
        });
    }
    
    createCentralAxis() {
        const axisGeometry = new THREE.CylinderGeometry(0.1, 0.1, 50);
        const axisMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x444444,
            transparent: true,
            opacity: 0.3
        });
        
        const axisMesh = new THREE.Mesh(axisGeometry, axisMaterial);
        axisMesh.position.y = 0;
        this.scene.add(axisMesh);
    }
    
    createStrandGuides() {
        const strandsArray = Object.keys(this.strandColors);
        const angleStep = (Math.PI * 2) / strandsArray.length;
        
        strandsArray.forEach((strandName, index) => {
            const points = [];
            const strandAngle = index * angleStep;
            
            for (let i = 0; i <= 100; i++) {
                const t = i / 100;
                const y = t * 30 - 15; // Height range
                const angle = strandAngle + t * Math.PI * 4; // 4 full rotations
                
                const x = this.helixRadius * Math.cos(angle);
                const z = this.helixRadius * Math.sin(angle);
                
                points.push(new THREE.Vector3(x, y, z));
            }
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({ 
                color: this.strandColors[strandName],
                transparent: true,
                opacity: 0.2
            });
            
            const line = new THREE.Line(geometry, material);
            line.userData = { type: 'guide', strand: strandName };
            this.scene.add(line);
        });
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Point lights for ambiance
        const pointLight1 = new THREE.PointLight(0x6366f1, 0.5, 50);
        pointLight1.position.set(15, 5, 15);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0x06b6d4, 0.5, 50);
        pointLight2.position.set(-15, 5, -15);
        this.scene.add(pointLight2);
    }
    
    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 10;
        this.controls.maxDistance = 100;
        this.controls.maxPolarAngle = Math.PI;
    }
    
    async loadBlockchainData() {
        try {
            const response = await fetch('http://localhost:8000/api/v1/blockchain/visualization');
            if (response.ok) {
                const data = await response.json();
                this.updateVisualization(data);
            }
        } catch (error) {
            console.log('Using simulated blockchain data for visualization');
            this.createSimulatedData();
        }
    }
    
    createSimulatedData() {
        // Create simulated blockchain data for demonstration
        const strands = Object.keys(this.strandColors);
        const simulatedData = {
            strands: {},
            cross_links: [],
            parameters: {
                radius: this.helixRadius,
                pitch: this.helixPitch,
                height: 20
            }
        };
        
        strands.forEach((strandName, strandIndex) => {
            const blockCount = Math.floor(Math.random() * 10) + 5;
            const strandData = [];
            
            for (let i = 0; i < blockCount; i++) {
                const angle = (strandIndex * Math.PI * 2 / strands.length) + (i * 0.3);
                const height = i * 2;
                const coherence = 0.3 + Math.random() * 0.7;
                
                strandData.push({
                    index: i,
                    x: this.helixRadius * Math.cos(angle),
                    y: height,
                    z: this.helixRadius * Math.sin(angle),
                    theta: angle,
                    radius: this.helixRadius,
                    coherence: coherence,
                    hash: `${strandName}_${i}`,
                    timestamp: new Date().toISOString()
                });
            }
            
            simulatedData.strands[strandName] = strandData;
        });
        
        // Create some cross-links
        for (let i = 0; i < 5; i++) {
            const sourceStrand = strands[Math.floor(Math.random() * strands.length)];
            const targetStrand = strands[Math.floor(Math.random() * strands.length)];
            
            if (sourceStrand !== targetStrand) {
                simulatedData.cross_links.push({
                    source: `${sourceStrand}_2`,
                    target: `${targetStrand}_2`,
                    source_strand: sourceStrand,
                    target_strand: targetStrand,
                    strength: Math.random(),
                    coherence_delta: Math.random() * 0.2
                });
            }
        }
        
        this.updateVisualization(simulatedData);
    }
    
    updateVisualization(data) {
        // Clear existing blocks and links
        this.clearVisualization();
        
        // Add blocks for each strand
        Object.entries(data.strands).forEach(([strandName, blocks]) => {
            this.addStrandBlocks(strandName, blocks);
        });
        
        // Add cross-links
        if (data.cross_links) {
            this.addCrossLinks(data.cross_links);
        }
        
        console.log('✅ Visualization updated with blockchain data');
    }
    
    addStrandBlocks(strandName, blocks) {
        const strandGroup = this.strandGroups[strandName];
        if (!strandGroup) return;
        
        blocks.forEach((block, index) => {
            // Create block geometry based on coherence
            const size = 0.3 + (block.coherence * 0.7);
            const geometry = new THREE.SphereGeometry(size, 16, 12);
            
            // Create material with strand color
            const color = this.strandColors[strandName];
            const material = new THREE.MeshLambertMaterial({ 
                color: color,
                transparent: true,
                opacity: 0.7 + (block.coherence * 0.3)
            });
            
            // Create mesh
            const blockMesh = new THREE.Mesh(geometry, material);
            blockMesh.position.set(block.x, block.y, block.z);
            blockMesh.userData = {
                type: 'block',
                strand: strandName,
                block: block,
                originalColor: color
            };
            
            // Add glow effect for high coherence blocks
            if (block.coherence > 0.8) {
                const glowGeometry = new THREE.SphereGeometry(size * 1.5, 16, 12);
                const glowMaterial = new THREE.MeshBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.1
                });
                const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
                glowMesh.position.copy(blockMesh.position);
                strandGroup.add(glowMesh);
            }
            
            strandGroup.add(blockMesh);
            this.blockMeshes.push(blockMesh);
            
            // Add block label
            this.addBlockLabel(blockMesh, block);
        });
    }
    
    addBlockLabel(blockMesh, block) {
        // Create text sprite for block hash
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 64;
        
        context.fillStyle = 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        context.fillStyle = '#ffffff';
        context.font = '16px Arial';
        context.textAlign = 'center';
        context.fillText(`#${block.index}`, canvas.width / 2, 25);
        context.fillText(block.hash.substring(0, 8), canvas.width / 2, 45);
        
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        
        sprite.position.copy(blockMesh.position);
        sprite.position.y += 1.5;
        sprite.scale.set(3, 0.75, 1);
        sprite.userData = { type: 'label', parent: blockMesh };
        
        this.scene.add(sprite);
    }
    
    addCrossLinks(crossLinks) {
        crossLinks.forEach(link => {
            // Find source and target blocks
            const sourceBlock = this.findBlockByHash(link.source);
            const targetBlock = this.findBlockByHash(link.target);
            
            if (sourceBlock && targetBlock) {
                // Create link geometry
                const points = [
                    sourceBlock.position.clone(),
                    targetBlock.position.clone()
                ];
                
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: 0xffffff,
                    transparent: true,
                    opacity: link.strength * 0.5,
                    linewidth: 2
                });
                
                const linkMesh = new THREE.Line(geometry, material);
                linkMesh.userData = {
                    type: 'cross_link',
                    link: link
                };
                
                this.scene.add(linkMesh);
                this.linkMeshes.push(linkMesh);
            }
        });
    }
    
    findBlockByHash(hash) {
        return this.blockMeshes.find(block => 
            block.userData.block && block.userData.block.hash === hash
        );
    }
    
    clearVisualization() {
        // Remove all blocks
        this.blockMeshes.forEach(block => {
            block.parent.remove(block);
        });
        this.blockMeshes = [];
        
        // Remove all links
        this.linkMeshes.forEach(link => {
            this.scene.remove(link);
        });
        this.linkMeshes = [];
        
        // Remove labels
        const labels = this.scene.children.filter(child => 
            child.userData && child.userData.type === 'label'
        );
        labels.forEach(label => this.scene.remove(label));
    }
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            this.time += 0.01;
            
            // Auto-rotation
            if (this.autoRotateEnabled) {
                this.scene.rotation.y += this.rotationSpeed;
            }
            
            // Animate block pulses based on coherence
            this.blockMeshes.forEach(block => {
                if (block.userData.block) {
                    const coherence = block.userData.block.coherence;
                    const pulse = 1 + Math.sin(this.time * 2 + coherence * 10) * 0.1 * coherence;
                    block.scale.setScalar(pulse);
                }
            });
            
            // Update controls
            this.controls.update();
            
            // Render
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    startAutoRotation() {
        this.autoRotateEnabled = true;
    }
    
    stopAutoRotation() {
        this.autoRotateEnabled = false;
    }
    
    resetView() {
        this.camera.position.set(25, 15, 25);
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }
    
    handleResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    async refresh() {
        // Reload blockchain data and update visualization
        await this.loadBlockchainData();
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        if (this.container && this.renderer) {
            this.container.removeChild(this.renderer.domElement);
        }
        
        window.removeEventListener('resize', this.handleResize);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.helixVisualization = new HelixVisualization();
});