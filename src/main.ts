import "./style.css";
// tsignore declaration error
import * as dat from 'dat.gui';

// TODO: how can we scale this up?

const GRID_SIZE = 512;
const NUM_STATES = 4;

let properties = {
    evolutionSpeed: 1,
    initialDensity: 0.3,
    raysPerCell: 8,
    maxBounces: 4,
    reinitialize: () => { }
};

function getInitComputeShaderCode(width: number, height: number) {
    return `
    struct Cell {
        state: u32,
        color: vec3<f32>,
    }

    @group(0) @binding(0) var<storage, read_write> cells: array<Cell>;

    const GRID_WIDTH: u32 = ${width};
    const GRID_HEIGHT: u32 = ${height};
    const NUM_STATES: u32 = ${NUM_STATES};
    const INITIAL_DENSITY: f32 = ${properties.initialDensity};

    fn hash(state: u32) -> u32 {
        var x = state;
        x = x ^ (x << 13u);
        x = x ^ (x >> 17u);
        x = x ^ (x << 5u);
        return x;
    }

    fn random(seed: u32) -> f32 {
        return f32(hash(seed)) / 4294967295.0;
    }

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.y * GRID_WIDTH + id.x;
        if (index >= GRID_WIDTH * GRID_HEIGHT) {
            return;
        }

        let seed = hash(index + id.x * 1237u + id.y * 3571u);
        
        if (random(seed) < INITIAL_DENSITY) {
            let state = 1u + (seed % (NUM_STATES - 1u));
            let hue = f32(state) / f32(NUM_STATES);
            let color = vec3<f32>(hue, 1.0, 0.8);  // HSV to RGB (simplified)
            cells[index] = Cell(state, color);
        } else {
            cells[index] = Cell(0u, vec3<f32>(0.0, 0.0, 0.0));
        }
    }
    `;
}

function getComputeShaderCode(width: number, height: number) {
    return `
struct Cell {
    state: u32,
    color: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;

const GRID_WIDTH: u32 = ${width};
const GRID_HEIGHT: u32 = ${height};
const NUM_STATES: u32 = ${NUM_STATES};
const RAYS_PER_CELL: u32 = ${properties.raysPerCell};
const MAX_BOUNCES: u32 = ${properties.maxBounces};

fn hash(state: u32) -> u32 {
    var x = state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    return x;
}

fn random(state: ptr<function, u32>) -> f32 {
    *state = hash(*state);
    return f32(*state) / 4294967295.0;
}

fn getCellIndex(x: u32, y: u32) -> u32 {
    return (y % GRID_HEIGHT) * GRID_WIDTH + (x % GRID_WIDTH);
}

fn evolveCell(index: u32) -> Cell {
    let x = index % GRID_WIDTH;
    let y = index / GRID_WIDTH;
    var neighborCount: u32 = 0;
    var newState: u32 = cells[index].state;

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let neighborX = (x + u32(dx) + GRID_WIDTH) % GRID_WIDTH;
            let neighborY = (y + u32(dy) + GRID_HEIGHT) % GRID_HEIGHT;
            let neighborIndex = getCellIndex(neighborX, neighborY);
            if (cells[neighborIndex].state > 0u) {
                neighborCount++;
            }
        }
    }

    // Simple cellular automata rules
    if (newState == 0u && neighborCount == 3u) {
        newState = 1u + (hash(index) % (NUM_STATES - 1u));
    } else if (newState > 0u && (neighborCount < 2u || neighborCount > 3u)) {
        newState = 0u;
    }

    // Generate color based on state
    let hue = f32(newState) / f32(NUM_STATES);
    let color = vec3<f32>(hue, 1.0, 0.8);  // HSV to RGB (simplified)

    return Cell(newState, color);
}

fn traceRay(origin: vec2<f32>, direction: vec2<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    var position = origin;
    var color = vec3<f32>(0.0);
    var attenuation = vec3<f32>(1.0);
    var directionmut = direction;

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let cellX = u32(position.x * f32(GRID_WIDTH));
        let cellY = u32(position.y * f32(GRID_HEIGHT));
        let cellIndex = getCellIndex(cellX, cellY);

        if (cells[cellIndex].state > 0u) {
            color += attenuation * cells[cellIndex].color;
            
            // Diffuse reflection
            let angle = random(seed) * 2.0 * 3.14159;
            directionmut = vec2<f32>(cos(angle), sin(angle));
            attenuation *= 0.5;
        }

        position += directionmut * vec2<f32>(1.0 / f32(GRID_WIDTH), 1.0 / f32(GRID_HEIGHT));

        if (position.x < 0.0 || position.x >= 1.0 || position.y < 0.0 || position.y >= 1.0) {
            break;
        }
    }

    return color;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.y * GRID_WIDTH + id.x;
    if (index >= GRID_WIDTH * GRID_HEIGHT) {
        return;
    }

    // Evolve cellular automata
    cells[index] = evolveCell(index);

    // Ray tracing
    var pixelColor = vec3<f32>(0.0);
    var seed = hash(index + id.x * 1237u + id.y * 3571u);

    for (var i = 0u; i < RAYS_PER_CELL; i++) {
        let origin = vec2<f32>(f32(id.x) / f32(GRID_WIDTH), f32(id.y) / f32(GRID_HEIGHT));
        let angle = random(&seed) * 2.0 * 3.14159;
        let direction = vec2<f32>(cos(angle), sin(angle));
        
        pixelColor += traceRay(origin, direction, &seed);
    }

    pixelColor /= f32(RAYS_PER_CELL);

    // Store the result
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(pixelColor, 1.0));
}
    `;
}


async function initWebGPU(canvas: HTMLCanvasElement) {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter?.requestDevice()!;
    const context = canvas.getContext("webgpu");
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    context?.configure({
        device,
        format: presentationFormat,
    });

    return { device, context, presentationFormat };
}

function createInitComputeShader(device: GPUDevice, cellsBuffer: GPUBuffer, textureSize: any) {
    const initModule = device.createShaderModule({
        label: "Cell Initialization Compute Shader",
        code: getInitComputeShaderCode(textureSize.width, textureSize.height)
    });

    const initPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: initModule,
            entryPoint: "main"
        }
    });

    const initBindGroup = device.createBindGroup({
        layout: initPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellsBuffer } }
        ]
    });

    return { initPipeline, initBindGroup };
}

function createComputeShader(device: GPUDevice, cellsBuffer: GPUBuffer, textureSize: any) {
    const module = device.createShaderModule({
        label: "Cellular Automata Compute Shader",
        code: getComputeShaderCode(textureSize.width, textureSize.height)
    });

    const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module,
            entryPoint: "main"
        }
    });

    const texture = device.createTexture({
        size: textureSize,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cellsBuffer } },
            { binding: 1, resource: texture.createView() }
        ]
    });

    return { pipeline, bindGroup, texture };
}

function createRenderPipeline(device: any, format: any) {
    const module = device.createShaderModule({
        label: "Render Shader",
        code: `
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

            @vertex
            fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                var pos = array<vec2<f32>, 4>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(1.0, -1.0),
                    vec2<f32>(-1.0, 1.0),
                    vec2<f32>(1.0, 1.0)
                );
                var uv = array<vec2<f32>, 4>(
                    vec2<f32>(0.0, 1.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(0.0, 0.0),
                    vec2<f32>(1.0, 0.0)
                );
                var output: VertexOutput;
                output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
                output.uv = uv[vertexIndex];
                return output;
            } 

            @group(0) @binding(0) var textureSampler: sampler;
            @group(0) @binding(1) var inputTexture: texture_2d<f32>;

            @fragment
            fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
                return textureSample(inputTexture, textureSampler, uv);
            }
        `
    });

    return device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vertexMain"
        },
        fragment: {
            module,
            entryPoint: "fragmentMain",
            targets: [{ format }]
        },
        primitive: {
            topology: "triangle-strip",
            stripIndexFormat: "uint32"
        }
    });
}
//
// function initializeCells(device: GPUDevice, width: number, height: number) {
//     console.log("width", width, "height", height);
//     const cellsData = new Uint32Array(width * height * 2);
//     for (let i = 0; i < cellsData.length; i += 2) {
//         if (Math.random() < properties.initialDensity) {
//             cellsData[i] = Math.floor(Math.random() * (NUM_STATES - 1)) + 1;
//         }
//     }
//
//     const cellsBuffer = device.createBuffer({
//         size: cellsData.byteLength,
//         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//     });
//
//     device.queue.writeBuffer(cellsBuffer, 0, cellsData);
//     return cellsBuffer;
// }

var textureSize = { width: GRID_SIZE, height: GRID_SIZE };

async function main() {
    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    const { device, context, presentationFormat } = await initWebGPU(canvas);
    textureSize = {
        width: Math.min(GRID_SIZE, canvas.width),
        height: Math.min(GRID_SIZE, canvas.height)
    };

    // let cellsBuffer = initializeCells(device, textureSize.width, textureSize.height);
    const cellsBuffer = device.createBuffer({
        size: textureSize.width * textureSize.height * 8, // 2 u32s per cell
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    let initComputeShader = createInitComputeShader(device, cellsBuffer, textureSize);
    const commandEncoder = device.createCommandEncoder();
    const initPass = commandEncoder.beginComputePass();
    initPass.setPipeline(initComputeShader.initPipeline);
    initPass.setBindGroup(0, initComputeShader.initBindGroup);
    initPass.dispatchWorkgroups(Math.ceil(textureSize.width / 16), Math.ceil(textureSize.height / 16));
    initPass.end();
    device.queue.submit([commandEncoder.finish()]);

    let computeShader = createComputeShader(device, cellsBuffer, textureSize);
    let renderPipeline = createRenderPipeline(device, presentationFormat);

    const sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
    });

    let renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: computeShader.texture.createView() }
        ]
    });

    function updateComputeShader(recreate = false) {
        if (recreate) {
            initComputeShader = createInitComputeShader(device, cellsBuffer, textureSize);
            const commandEncoder = device.createCommandEncoder();
            const initPass = commandEncoder.beginComputePass();
            initPass.setPipeline(initComputeShader.initPipeline);
            initPass.setBindGroup(0, initComputeShader.initBindGroup);
            initPass.dispatchWorkgroups(Math.ceil(textureSize.width / 16), Math.ceil(textureSize.height / 16));
            initPass.end();
            device.queue.submit([commandEncoder.finish()]);
        }
        computeShader = createComputeShader(device, cellsBuffer, textureSize);
        renderBindGroup = device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: sampler },
                { binding: 1, resource: computeShader.texture.createView() }
            ]
        });
    }

    properties.reinitialize = () => updateComputeShader(true);

    function render() {
        const commandEncoder = device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(computeShader.pipeline);
        computePass.setBindGroup(0, computeShader.bindGroup);
        computePass.dispatchWorkgroups(Math.ceil(textureSize.width / 16), Math.ceil(textureSize.height / 16));
        computePass.end();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context!.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.draw(4);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    function animate() {
        for (let i = 0; i < properties.evolutionSpeed; i++) {
            render();
        }
        requestAnimationFrame(animate);
    }

    animate();

    // GUI setup
    const gui = new dat.GUI();
    gui.add(properties, 'evolutionSpeed', 1, 10).step(1);
    gui.add(properties, 'initialDensity', 0, 1).step(0.05).onChange(() => updateComputeShader(true));
    gui.add(properties, 'raysPerCell', 1, 32).step(1).onChange(() => updateComputeShader(true));
    gui.add(properties, 'maxBounces', 1, 10).step(1).onChange(() => updateComputeShader(true));
    gui.add(properties, 'reinitialize');

    // ResizeObserver setup
    const observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            const width = entry.devicePixelContentBoxSize?.[0].inlineSize ||
                entry.contentBoxSize[0].inlineSize * devicePixelRatio;
            const height = entry.devicePixelContentBoxSize?.[0].blockSize ||
                entry.contentBoxSize[0].blockSize * devicePixelRatio;

            canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
            canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));

            // Update context configuration
            context?.configure({
                device,
                format: presentationFormat,
            });

            // Update texture size
            textureSize = {
                width: Math.min(GRID_SIZE, canvas.width),
                height: Math.min(GRID_SIZE, canvas.height)
            };

            // Recreate compute shader with new size
            updateComputeShader(true);

            // Re-render
            render();
        }
    });

    try {
        observer.observe(canvas, { box: 'device-pixel-content-box' });
    } catch {
        observer.observe(canvas, { box: 'content-box' });
    }
}

main().catch(console.error);
