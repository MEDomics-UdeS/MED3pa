// Color Interpolation Function
function interpolateColor(color1, color2, factor) {
    const hex = (color) => parseInt(color.slice(1), 16);
    const r = (color) => (color >> 16) & 255;
    const g = (color) => (color >> 8) & 255;
    const b = (color) => color & 255;

    const c1 = hex(color1);
    const c2 = hex(color2);
    const rVal = Math.round(r(c1) + factor * (r(c2) - r(c1)));
    const gVal = Math.round(g(c1) + factor * (g(c2) - g(c1)));
    const bVal = Math.round(b(c1) + factor * (b(c2) - b(c1)));
    
    return `rgb(${rVal}, ${gVal}, ${bVal})`;
}

// Determine color range dynamically based on parameter
function getColorForValue(value, max, min) {
    const range = max - min;
    const normalizedValue = Math.min(Math.max((value - min) / range, 0), 1);
    let color;

    if (normalizedValue < 0.5) {
        color = interpolateColor("#c90404", "#e07502", normalizedValue * 2); // Red to Orange
    } else {
        color = interpolateColor("#e07502", "#068a0c", (normalizedValue - 0.5) * 2); // Orange to Green
    }
    
    return color;
}

function toggleLegendVisibility(show) {
    const legendContainer = document.querySelector(".legend-container");
    legendContainer.style.display = show ? "flex" : "none";
}


// Apply color to nodes based on parameter and custom min/max range
function applyColorToNodes() {
    const infoType = document.querySelector("input[name='info-type']:checked").value;
    const parameter = document.getElementById("color-parameter").value;
    if (parameter === "") return; // Exit if no parameter is selected

    const min = parseFloat(document.getElementById("color-min").value);
    const max = parseFloat(document.getElementById("color-max").value);

    treeData.forEach((node) => {
        const nodeElement = document.getElementById(`node-${node.id}`);
        if (node[infoType] && node[infoType][parameter] != null && nodeElement) {
            const value = node[infoType][parameter];
            const color = getColorForValue(value, max, min);

            const titleElement = nodeElement.querySelector('.node-title');
            titleElement.style.backgroundColor = color;
        }
    });
    toggleLegendVisibility(true);
}

// Reset Node Colors
function resetNodeColors() {
    treeData.forEach((node) => {
        const nodeElement = document.getElementById(`node-${node.id}`);
        if (nodeElement) {
            const titleElement = nodeElement.querySelector('.node-title');
            titleElement.style.backgroundColor = ""; // Reset color
        }
    });
    toggleLegendVisibility(false)
}

// Function to create node content
function createNodeContent(node, infoType, isPhantom = false) {
    const container = document.createElement("div");
    container.className = "node-container";

    const title = document.createElement("div");
    title.className = isPhantom ? "node-title lost-profile-title" : "node-title";
    title.innerText = isPhantom ? "Lost Profile" : `Profile ${node.id}`;
    container.appendChild(title);

    const content = document.createElement("div");
    content.className = "node-content";

    const info = node[infoType];

    if (info && !isPhantom) {
        if (infoType === "detectron_results") {
            // Handle detectron_results specifically
            if (info["Tests Results"]) {
                content.innerHTML = info["Tests Results"]
                    .map((result) => {
                        // Extract relevant information
                        const strategy = `Strategy: ${result.Strategy}`;
                        const shiftProbOrPValue =
                            result.shift_probability !== undefined
                                ? `Shift Probability: ${result.shift_probability.toFixed(4)}`
                                : result.p_value !== undefined
                                ? `P-Value: ${result.p_value.toFixed(4)}`
                                : "No data available";
                        return `${strategy}<br>${shiftProbOrPValue}`;
                    })
                    .join("<br><br>"); // Add spacing between strategies
            } else {
                content.innerText = "Detectron was not executed!";
            }
        } else {
            // Default case for other info types
            content.innerHTML = Object.entries(info)
                .map(([key, value]) => {
                    const formattedValue =
                        typeof value === "number" && !Number.isInteger(value)
                            ? value.toFixed(4)
                            : value;
                    return `${key}: ${formattedValue !== null ? formattedValue : "N/A"}`;
                })
                .join("<br>");
        }
    } else {
        content.innerText = "No data available";
    }

    container.appendChild(content);
    return container;
}


// Build the tree
// Build the tree and re-render nodes based on selected info type
function buildTree(data, rootElement, parentPath = ["*"], infoType = "node information") {
    const rootNode = data.find(node => JSON.stringify(node.path) === JSON.stringify(parentPath));
    if (rootNode) {
        const li = document.createElement("li");

        const nodeContent = createNodeContent(rootNode, infoType);
        nodeContent.id = `node-${rootNode.id}`;
        li.appendChild(nodeContent);

        const rootUl = document.createElement("ul");
        li.appendChild(rootUl);
        rootElement.appendChild(li);

        buildChildren(data, rootUl, rootNode.path, infoType);
    }
}

// Recursively build children nodes based on selected info type
function buildChildren(data, parentElement, parentPath, infoType) {
    const children = data.filter(node => JSON.stringify(node.path.slice(0, -1)) === JSON.stringify(parentPath));

    children.forEach((node) => {
        const li = document.createElement("li");

        const conditionLabel = document.createElement("div");
        conditionLabel.className = "condition-label";
        conditionLabel.innerText = node.path[node.path.length - 1];
        li.appendChild(conditionLabel);

        const nodeContent = createNodeContent(node, infoType);
        nodeContent.id = `node-${node.id}`;
        li.appendChild(nodeContent);

        parentElement.appendChild(li);

        const hasChildren = data.some(n => JSON.stringify(n.path.slice(0, -1)) === JSON.stringify(node.path));
        if (hasChildren) {
            const ul = document.createElement("ul");
            buildChildren(data, ul, node.path, infoType);
            li.appendChild(ul);
        }
    });

    if (children.length === 1) {
        const phantomLi = document.createElement("li");

        const existingCondition = children[0].path[children[0].path.length - 1];
        const oppositeCondition = existingCondition.includes("<=")
            ? existingCondition.replace("<=", ">")
            : existingCondition.replace(">", "<=");

        const phantomConditionLabel = document.createElement("div");
        phantomConditionLabel.className = "condition-label";
        phantomConditionLabel.innerText = oppositeCondition;
        phantomLi.appendChild(phantomConditionLabel);

        const phantomNodeContent = createNodeContent({}, infoType, true);
        phantomLi.appendChild(phantomNodeContent);

        parentElement.appendChild(phantomLi);
    }
}

// Update tree display based on selected information type
function updateTreeDisplay() {
    const infoType = document.querySelector("input[name='info-type']:checked").value;
    const treeRoot = document.getElementById("tree-root");
    treeRoot.innerHTML = ""; // Clear existing tree
    buildTree(treeData, treeRoot, ["*"], infoType); // Rebuild tree with selected info type
}

// Update options based on selected info type
function updateColorParameterOptions(infoType) {
    const colorParameterSelect = document.getElementById("color-parameter");
    colorParameterSelect.innerHTML = "<option value=''>Select Parameter</option>";

    const sampleNode = treeData.find(node => node[infoType]);
    if (sampleNode && sampleNode[infoType]) {
        Object.keys(sampleNode[infoType]).forEach(key => {
            const option = document.createElement("option");
            option.value = key;
            option.textContent = key;
            colorParameterSelect.appendChild(option);
        });
    }

    document.getElementById("color-min").value = infoType === "metrics" ? 0 : 0;
    document.getElementById("color-max").value = infoType === "metrics" ? 1 : 100;
}

// Disable checkboxes based on available data
function updateCheckboxAvailability() {
    console.log(document.getElementById("general-info-checkbox"));
    document.getElementById("general-info-checkbox").disabled = !treeData.some(node => node["node information"]);
    document.getElementById("performance-info-checkbox").disabled = !treeData.some(node => node.metrics);
    document.getElementById("shift-detection-checkbox").disabled = !treeData.some(node => node.detectron_results);
}

// Initialize the Tree
function initializeTree() {
    const treeRoot = document.getElementById("tree-root");
    buildTree(treeData, treeRoot);
    console.log(treeData)
    updateCheckboxAvailability();
    updateColorParameterOptions("node information"); 
}

const treeContainer = document.getElementById("tree-root");
panzoom(treeContainer);

function downloadTreeAsPNG() {
    const treeContainer = document.getElementById("tree-container");
    
    htmlToImage.toPng(treeContainer)
        .then((dataUrl) => {
            const link = document.createElement("a");
            link.href = dataUrl;
            link.download = "tree.png";
            link.click();
        })
        .catch((error) => {
            console.error("Error generating PNG with html-to-image: ", error);
        });
}

function updateLegendLabels() {
    const minRangeInput = document.getElementById("color-min");
    const maxRangeInput = document.getElementById("color-max");

    // Convert values to numbers
    const minRange = parseFloat(minRangeInput.value);
    const maxRange = parseFloat(maxRangeInput.value);

    console.log(typeof minRange)
    document.getElementById("min-legend").textContent = minRange;
    document.getElementById("quarter-legend").textContent = ((maxRange - minRange) * 0.25 + minRange).toFixed(2);
    document.getElementById("half-legend").textContent = ((maxRange - minRange) * 0.5 + minRange).toFixed(2);
    document.getElementById("three-quarters-legend").textContent = ((maxRange - minRange) * 0.75 + minRange).toFixed(2);
    document.getElementById("max-legend").textContent = maxRange;
}

function toggleColorSection() {
    const colorOptions = document.getElementById("color-options");
    const colorToggle = document.getElementById("color-toggle");

    if (colorToggle.checked) {
        colorOptions.style.display = "block";
    } else {
        colorOptions.style.display = "none";
    }
}

// Event listeners to update legend when min or max range changes
document.getElementById("color-min").addEventListener("input", updateLegendLabels);
document.getElementById("color-max").addEventListener("input", updateLegendLabels);

// Initial call to set legend values on page load
updateLegendLabels();


// Event listeners
document.getElementById("color-nodes-button").addEventListener("click", applyColorToNodes);
document.getElementById("reset-color-button").addEventListener("click", resetNodeColors);
document.getElementById("download-png-button").addEventListener("click", downloadTreeAsPNG);
document.querySelectorAll("input[name='info-type']").forEach((radio) => {
    radio.addEventListener("change", (event) => {
        updateColorParameterOptions(event.target.value);
        updateLegendLabels();
        updateTreeDisplay()
    });
});

function updateColorToggleAvailability() {
    const infoType = document.querySelector("input[name='info-type']:checked").value;
    const colorToggle = document.getElementById("color-toggle");
    const colorToggleContainer = document.querySelector(".color-toggle");

    if (infoType === "detectron_results") {
        // Disable the toggle and hide the color options
        colorToggle.checked = false;
        colorToggle.disabled = true;
        colorToggleContainer.classList.add("disabled-toggle");
        document.getElementById("color-options").style.display = "none";
    } else {
        // Enable the toggle
        colorToggle.disabled = false;
        colorToggleContainer.classList.remove("disabled-toggle");
    }
}

// Attach this function to the event listener for radio buttons
document.querySelectorAll("input[name='info-type']").forEach((radio) => {
    radio.addEventListener("change", updateColorToggleAvailability);
});

// Initial call to ensure toggle is correctly enabled/disabled on page load
updateColorToggleAvailability();

document.addEventListener("DOMContentLoaded", function() {
    initializeTree();
});

