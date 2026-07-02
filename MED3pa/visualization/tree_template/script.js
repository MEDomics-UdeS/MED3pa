// Names to change
const replacements = {
    "service_group_": "",
    "admission_group_": "admission_",
    "&lt;": "<",
    "&gt;": ">"
};

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
    title.innerText = isPhantom ? "Lost Profile" : `Profile ${node.node_id}`;
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
                                ? `Shift Probability: ${result.shift_probability.toFixed(2)}`
                                : result.p_value !== undefined
                                ? `P-Value: ${result.p_value.toFixed(2)}`
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
                            ? value.toFixed(2)
                            : value;
                    const color = node.text_color && key in node.text_color ? node.text_color[key] : "black";
                    if (node.highlight) {
                        content.style.fontSize = "20px";
                    }
                    return `<p style="color:${color};">${key}: ${formattedValue !== null ? formattedValue : "N/A"}</p>`;
                })
                .join("");

            // content.innerHTML = Object.entries(info)
            //     .map(([key, value]) => {
            //         const formattedValue =
            //             typeof value === "number" && !Number.isInteger(value)
            //                 ? value.toFixed(3)
            //                 : value;
            //         return `${key}: ${formattedValue !== null ? formattedValue : "N/A"}`;
            //     })
            //     .join("<br>");
        }
    } else {
        content.innerText = "No data available";
    }

    container.appendChild(content);
    return container;
}


// Build the tree
// Build the tree and re-render nodes based on selected info type
function buildTree(data, rootElement, parentPath = ["*"], infoType = "node_information") {
    const rootNode = data.find(node => JSON.stringify(node.path) === JSON.stringify(parentPath));
    if (rootNode) {
        const li = document.createElement("li");

        // const conditionLabel = document.createElement("div");
        // conditionLabel.className = "condition-label";
        // conditionLabel.innerText = '*';
        // li.appendChild(conditionLabel);

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
    // console.log(document.getElementById("general-info-checkbox"));
    document.getElementById("general-info-checkbox").disabled = !treeData.some(node => node["node_information"]);
    document.getElementById("performance-info-checkbox").disabled = !treeData.some(node => node.metrics);
    document.getElementById("shift-detection-checkbox").disabled = !treeData.some(node => node.detectron_results);
}

// Initialize the Tree
function initializeTree() {
    const treeRoot = document.getElementById("tree-root");
    buildTree(treeData, treeRoot);
    // console.log(treeData)
    updateCheckboxAvailability();
    updateColorParameterOptions("node_information");
}

const treeContainer = document.getElementById("tree-root");
panzoom(treeContainer);

// function downloadTreeAsPNG() {
//     const treeContainer = document.getElementById("tree-container");
//
//     htmlToImage.toPng(treeContainer)
//         .then((dataUrl) => {
//             const link = document.createElement("a");
//             link.href = dataUrl;
//             link.download = "tree.png";
//             link.click();
//         })
//         .catch((error) => {
//             console.error("Error generating PNG with html-to-image: ", error);
//         });
// }

function downloadTreeAsPDF() {
    const treeContainer = document.getElementById("tree-container");

    // Ensure domtoimage is loaded
    if (typeof window.domtoimage === "undefined") {
        console.error("dom-to-image-more is not loaded!");
        return;
    }

    window.domtoimage.toPng(treeContainer) // Convert to PNG for embedding in PDF
        .then((dataUrl) => {
            const pdf = new window.jspdf.jsPDF(); // Create a new PDF document
            const imgWidth = 190; // Width in mm
            const imgHeight = (treeContainer.clientHeight / treeContainer.clientWidth) * imgWidth; // Maintain aspect ratio

            pdf.addImage(dataUrl, "PNG", 10, 10, imgWidth, imgHeight);
            pdf.save("tree.pdf"); // Download as PDF
        })
        .catch((error) => {
            console.error("Error generating PDF with dom-to-image-more: ", error);
        });
}

// function downloadTreeAsSVG() {
//     const treeContainer = document.getElementById("tree-container");
//
//     // Ensure domtoimage is loaded
//     if (typeof window.domtoimage === "undefined") {
//         console.error("dom-to-image-more is not loaded!");
//         return;
//     }
//
//     window.domtoimage.toSvg(treeContainer)
//         .then((dataUrl) => {
//             const link = document.createElement("a");
//             link.href = dataUrl;
//             link.download = "tree.svg";
//             link.click();
//         })
//         .catch((error) => {
//             console.error("Error generating SVG with dom-to-image-more: ", error);
//         });
// }

function downloadTreeAsSVG() {
    const treeContainer = document.getElementById("tree-container");

    if (!treeContainer) {
        console.error("Tree container not found!");
        return;
    }

    const rect = treeContainer.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);

    // Create the SVG element
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");

    // Background
    const bgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bgRect.setAttribute("width", "100%");
    bgRect.setAttribute("height", "100%");
    bgRect.setAttribute("fill", "white");
    svg.appendChild(bgRect);

    const nodes = treeContainer.querySelectorAll(".node-container");
    let positions = {};
    let bg_colors = {};

    // Zoom factor
    let zoomFactor = parseFloat(document.getElementById("tree-root").style.transform.match(/matrix\(([^)]+)\)/)[1].split(',')[0]);

    // Define round rectangles
    const cornerRadius = 5 * zoomFactor; // Scale based on zoom

    const clipPath = `<clipPath id="rounded-top">
        <path d="M${cornerRadius},0 h${width - 2 * cornerRadius}
        a${cornerRadius},${cornerRadius} 0 0 1 ${cornerRadius},${cornerRadius}
        v${height - cornerRadius} h-${width} v-${height - cornerRadius}
        a${cornerRadius},${cornerRadius} 0 0 1 ${cornerRadius},-${cornerRadius} z"/>
    </clipPath>`;

    // Create an SVG filter for the shadow around rectangles
    const svgFilter = document.createElementNS("http://www.w3.org/2000/svg", "filter");
    svgFilter.setAttribute("id", "shadow");
    svgFilter.setAttribute("x", "-20%");
    svgFilter.setAttribute("y", "-20%");
    svgFilter.setAttribute("width", "140%");
    svgFilter.setAttribute("height", "140%");

    // Create a Gaussian blur effect for the shadow
    const feGaussianBlur = document.createElementNS("http://www.w3.org/2000/svg", "feGaussianBlur");
    feGaussianBlur.setAttribute("in", "SourceAlpha");
    feGaussianBlur.setAttribute("stdDeviation", 3 * zoomFactor); // Adjust blur size
    svgFilter.appendChild(feGaussianBlur);

    // Merge original shape with shadow
    const feMerge = document.createElementNS("http://www.w3.org/2000/svg", "feMerge");
    const feMergeNode1 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
    // feMergeNode1.setAttribute("in", "offsetBlur");
    feMerge.appendChild(feMergeNode1);
    const feMergeNode2 = document.createElementNS("http://www.w3.org/2000/svg", "feMergeNode");
    feMergeNode2.setAttribute("in", "SourceGraphic");
    feMerge.appendChild(feMergeNode2);

    svgFilter.appendChild(feMerge);

    // Append filter to SVG defs
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.appendChild(svgFilter);
    svg.appendChild(defs);

    // First pass: Draw nodes, titles, and content
    nodes.forEach((node, index) => {
        let nodeId;
        if (node.id === ""){
            const parentNode = node.closest("li").parentElement.closest("li")?.querySelector(".node-container");
            nodeId = 'child_' + parentNode.id;
        }
        else {
            nodeId = node.id;
        }
        const nodeRect = node.getBoundingClientRect();
        const x = nodeRect.left - rect.left;
        const y = nodeRect.top - rect.top;
        const width = nodeRect.width;
        const height = nodeRect.height;
        positions[nodeId] = { x, y, width, height };

        // Get background color from .node-title
        const titleElement = node.querySelector(".node-title");
        let backgroundColor = "white"; // Default
        if (titleElement) {
            const computedStyle = window.getComputedStyle(titleElement);
            backgroundColor = computedStyle.backgroundColor || "white";
        }
        bg_colors[nodeId] = backgroundColor

        // Draw node rectangle with white background color
        const svgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        svgRect.setAttribute("x", x);
        svgRect.setAttribute("y", y);
        svgRect.setAttribute("rx", 5 * zoomFactor);
        svgRect.setAttribute("ry", 5 * zoomFactor);
        svgRect.setAttribute("width", width);
        svgRect.setAttribute("height", height);
        svgRect.setAttribute("fill", "white");
        svgRect.setAttribute("stroke", "black");
        svgRect.setAttribute("stroke-width", "0.25")
        svgRect.setAttribute("filter", "url(#shadow)"); // Apply the shadow filter
        svgRect.setAttribute("style", "opacity: 0.5;")
        svg.appendChild(svgRect);

        // **If it's the first node, add the condition rectangle ("*")**
        if (index === 0) {
            // const conditionHeight = 20 * zoomFactor; // Height of the condition box
            // const conditionRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            // conditionRect.setAttribute("x", x);
            // conditionRect.setAttribute("y", y);
            // conditionRect.setAttribute("rx", 5 * zoomFactor);
            // conditionRect.setAttribute("ry", 5 * zoomFactor);
            // conditionRect.setAttribute("width", width);
            // conditionRect.setAttribute("height", conditionHeight);
            // conditionRect.setAttribute("fill", bg_colors[nodeId]); // Match background color
            // conditionRect.setAttribute("stroke", "black");
            // svg.appendChild(conditionRect);
            // Define a path with rounded top corners only
            const clipRect = document.createElementNS("http://www.w3.org/2000/svg", "path");
            const r = 5 * zoomFactor; // Radius for top corners
            const w = width;
            const h = 20 * zoomFactor; // Height of the condition box
            const conditionPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
            conditionPath.setAttribute("d", `M${x + r},${y} 
                                            H${x + w - r} 
                                            A${r},${r} 0 0 1 ${x + w},${y + r} 
                                            V${y + h} 
                                            H${x} 
                                            V${y + r} 
                                            A${r},${r} 0 0 1 ${x + r},${y} 
                                            Z`);
            conditionPath.setAttribute("fill", bg_colors[nodeId]);
            conditionPath.setAttribute("stroke", "black");
            conditionPath.setAttribute("stroke-width", "0.25")
            svg.appendChild(conditionPath);

            // // Add "*" text inside the condition rectangle
            // const conditionText = document.createElementNS("http://www.w3.org/2000/svg", "text");
            // conditionText.setAttribute("x", x + width / 2);
            // conditionText.setAttribute("y", y + conditionHeight / 2 + 5*zoomFactor);
            // conditionText.setAttribute("font-size", 14 * zoomFactor);
            // conditionText.setAttribute("fill", bg_colors[node.id] === "rgb(49, 49, 49)" ? "white" : "black");
            // conditionText.setAttribute("text-anchor", "middle");
            // conditionText.setAttribute("dominant-baseline", "middle");
            // conditionText.textContent = "*";
            // svg.appendChild(conditionText);
        }

        // Get node content and handle <p> elements
        const contentElement = node.querySelector(".node-content");
        if (contentElement) {
            let paragraphs = contentElement.querySelectorAll("p"); // Select all <p> elements
            let fontsize = parseFloat(window.getComputedStyle(contentElement).fontSize) || 11;
            let lineHeight = (fontsize * 1.33) * zoomFactor;
            let startY = y + lineHeight + 20 * zoomFactor;//+ height / 2 - (paragraphs.length) * (lineHeight / 2) + 20 * zoomFactor;

            paragraphs.forEach((p, i) => {
                const contentText = document.createElementNS("http://www.w3.org/2000/svg", "text");
                contentText.setAttribute("x", x + 10 * zoomFactor);
                contentText.setAttribute("y", startY + i * lineHeight);
                // console.log(contentElement.style.fontSize);
                contentText.setAttribute("font-size", fontsize * zoomFactor);

                // Extract color from <p> style
                let textColor = p.style.color || "black";
                if (textColor === "black"){
                    contentText.setAttribute("fill", "black")
                }
                else {
                    contentText.setAttribute("fill", "white");
                }
                contentText.setAttribute("text-anchor", "left");
                contentText.textContent = p.textContent.trim();

                svg.appendChild(contentText);

                if (textColor !== "black"){
                    // Append text to SVG temporarily to measure size
                    const textBBox = contentText.getBBox(); // Get actual width & height

                    // Use getComputedTextLength() to get text width
                    let textWidth = p.clientWidth;// contentText.getComputedTextLength();
                    let textHeight = fontsize * zoomFactor * 1.2; // Approximate height

                    // Create a black rectangle behind the text
                    const backgroundRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                    backgroundRect.setAttribute("x", x + 5 * zoomFactor); // Slightly offset
                    backgroundRect.setAttribute("y", startY + i * lineHeight - textHeight * 0.8); // Align with text
                    backgroundRect.setAttribute("width", (textWidth + 10) * zoomFactor); // Add padding
                    backgroundRect.setAttribute("height", textHeight); // Adjust height slightly
                    backgroundRect.setAttribute("fill", textColor);
                    backgroundRect.setAttribute("rx", 4 * zoomFactor); // Rounded corners (optional)

                    // Move rect before text to layer it behind
                    svg.insertBefore(backgroundRect, contentText);


                    // // Add rectangle behind text
                    //  // Measure text width
                    // let textWidth = p.textContent.length * fontsize * 0.5 * zoomFactor; // Approximation
                    // let textHeight = fontsize * zoomFactor * 1.2; // Slightly larger than font-size
                    // // Create a black rectangle behind the text
                    // const backgroundRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                    // backgroundRect.setAttribute("x", x + 5 * zoomFactor); // Slightly offset
                    // backgroundRect.setAttribute("y", startY + i * lineHeight - textHeight * 0.8); // Align with text
                    // backgroundRect.setAttribute("width", textWidth);
                    // backgroundRect.setAttribute("height", textHeight);
                    // backgroundRect.setAttribute("fill", textColor);
                    // backgroundRect.setAttribute("rx", 4 * zoomFactor); // Rounded corners (optional)
                    // // Append rectangle before text (so it's behind)
                    // svg.appendChild(backgroundRect);
                }

            });
        }

        // // Get node content and handle <br>
        // const contentElement = node.querySelector(".node-content");
        // if (contentElement) {
        //     let contentLines = contentElement.innerHTML.split(/<br\s*\/?>/i);
        //     let lineHeight = 14 * zoomFactor;
        //     let startY = y + height / 2 - (contentLines.length ) * (lineHeight / 2) + 20 * zoomFactor;
        //
        //     contentLines.forEach((line, i) => {
        //         const contentText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        //         contentText.setAttribute("x", x + 10 * zoomFactor);
        //         contentText.setAttribute("y", startY + i * lineHeight);
        //         contentText.setAttribute("font-size", 12 * zoomFactor);
        //         contentText.setAttribute("fill", "black");
        //         contentText.setAttribute("text-anchor", "left");
        //         contentText.textContent = line.trim();
        //         svg.appendChild(contentText);
        //     });
        // }
    });

    // Second pass: Draw connections and condition labels
    const conditions = treeContainer.querySelectorAll(".condition-label");
    conditions.forEach((conditionLabel) => {
        const parentNode = conditionLabel.closest("li").parentElement.closest("li")?.querySelector(".node-container");
        const childNode = conditionLabel.closest("li")?.querySelector(".node-container");

        if (!parentNode || !childNode) return;

        const parentID = parentNode.id;
        let childID;
        if (childNode.id === ""){
            childID = 'child_' + parentNode.id;
        }
        else {
            childID = childNode.id;
        }
        if (positions[parentID] && positions[childID]) {
            const { x: px, y: py, width: pw, height: ph } = positions[parentID];
            const { x: cx, y: cy, width: cw, height: ch } = positions[childID];

            const x1 = px + pw / 2;
            const y1 = py + ph;
            const x2 = cx + cw / 2;
            const y2 = cy;

            const midX = (x1 + x2) / 2; // Control point in the middle
            const midY = (y1 + y2) / 2; // Adjust curve height (negative = curve upwards)
            const controlX1 = x1; // Control points for first downward curve
            const controlY1 = y1 + 50;

            const controlX2 = midX; // Control points for final downward curve
            const controlY2 = y2 - 50;

            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", `M ${x1},${y1} 
                            C ${x1},${y2} ${x2},${y1} ${x2},${y2}`);
            path.setAttribute("stroke", "black");
            path.setAttribute("fill", "none");
            path.setAttribute("stroke-width", 2 * zoomFactor);
            svg.appendChild(path)

            // // Draw line from parent to child
            // const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            // line.setAttribute("x1", px + pw / 2);
            // line.setAttribute("y1", py + ph);
            // line.setAttribute("x2", cx + cw / 2);
            // line.setAttribute("y2", cy);
            // line.setAttribute("stroke", "black");
            // line.setAttribute("stroke-width", "1");
            // svg.appendChild(line);

            // Add condition label inside a small rectangle
            let conditionText = conditionLabel.innerText.trim();
            // Loop through the dictionary and replace occurrences
            for (const [key, value] of Object.entries(replacements)) {
                conditionText = conditionText.replace(new RegExp(key, "g"), value);
            }
            // conditionText = conditionText.replace('service_group_', '').replace('admission_group_','admission_')
            console.log(conditionText)
            const conditionRectWidth = cw;
            const conditionRectHeight = 20 * zoomFactor;
            const condX = cx;
            const condY = cy ;  // - conditionRectHeight

            // // Small rectangle for the condition label
            // const conditionRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            // conditionRect.setAttribute("x", condX);
            // conditionRect.setAttribute("y", condY);
            // conditionRect.setAttribute("rx", 5 * zoomFactor);
            // conditionRect.setAttribute("ry", 5 * zoomFactor);
            // conditionRect.setAttribute("width", conditionRectWidth);
            // conditionRect.setAttribute("height", conditionRectHeight);
            // conditionRect.setAttribute("fill", bg_colors[childID]);
            // conditionRect.setAttribute("stroke", "black");
            // svg.appendChild(conditionRect);

            // Define a path with rounded top corners only
            const clipRect = document.createElementNS("http://www.w3.org/2000/svg", "path");
            const r = 5 * zoomFactor; // Radius for top corners
            const w = conditionRectWidth;
            const h = conditionRectHeight;
            const conditionPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
            conditionPath.setAttribute("d", `M${condX + r},${condY} 
                                            H${condX + w - r} 
                                            A${r},${r} 0 0 1 ${condX + w},${condY + r} 
                                            V${condY + h} 
                                            H${condX} 
                                            V${condY + r} 
                                            A${r},${r} 0 0 1 ${condX + r},${condY} 
                                            Z`);
            conditionPath.setAttribute("fill", bg_colors[childID]);
            conditionPath.setAttribute("stroke", "black");
            conditionPath.setAttribute("stroke-width", "0.25")
            svg.appendChild(conditionPath);

            // Condition label text inside the small rectangle
            const conditionTextElement = document.createElementNS("http://www.w3.org/2000/svg", "text");
            conditionTextElement.setAttribute("x", condX + conditionRectWidth / 2);
            conditionTextElement.setAttribute("y", condY + conditionRectHeight / 2 + 5*zoomFactor);
            conditionTextElement.setAttribute("font-size", 14 * zoomFactor);
            if (bg_colors[childID] === "rgb(49, 49, 49)"){
                conditionTextElement.setAttribute("fill", "white");
            }
            else {
                conditionTextElement.setAttribute("fill", "black");
            }
            conditionTextElement.setAttribute("text-anchor", "middle");
            conditionTextElement.textContent = conditionText;
            console.log(conditionTextElement)
            svg.appendChild(conditionTextElement);
        }
    });

    // Serialize and download
    const serializer = new XMLSerializer();
    let svgString = serializer.serializeToString(svg);
    // // Ensure correct encoding for `<` and `>`
    // svgString = svgString.replace(/&lt;/g, "\\textless").replace(/&gt;/g, "\\textgreater");
    const blob = new Blob([svgString], { type: "image/svg+xml" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "tree.svg";
    link.click();
}









function updateLegendLabels() {
    const minRangeInput = document.getElementById("color-min");
    const maxRangeInput = document.getElementById("color-max");

    // Convert values to numbers
    const minRange = parseFloat(minRangeInput.value);
    const maxRange = parseFloat(maxRangeInput.value);

    // console.log(typeof minRange)
    document.getElementById("min-legend").textContent = minRange;
    document.getElementById("quarter-legend").textContent = ((maxRange - minRange) * 0.25 + minRange).toFixed(3);
    document.getElementById("half-legend").textContent = ((maxRange - minRange) * 0.5 + minRange).toFixed(3);
    document.getElementById("three-quarters-legend").textContent = ((maxRange - minRange) * 0.75 + minRange).toFixed(3);
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
document.getElementById("download-png-button").addEventListener("click", downloadTreeAsSVG);
document.getElementById("download-pdf-button").addEventListener("click", downloadTreeAsPDF);
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

// document.addEventListener("DOMContentLoaded", function () {
//     const performanceRadio = document.getElementById("performance-info-checkbox");
//     const metricsFilter = document.getElementById("metrics-filter");
//     const metricsSelect = document.getElementById("metrics-select");
//
//     const availableMetrics = ["Accuracy", "Auc", "Auprc", "BalancedAccuracy", "F1Score", "LogLoss",
//     "MCC", "NPV", "PPV", "Precision", "Recall", "Sensitivity", "Specificity"]; // Example metrics
//
//     // Populate dropdown
//     availableMetrics.forEach(metric => {
//         const option = document.createElement("option");
//         option.value = metric;
//         option.textContent = metric;
//         metricsSelect.appendChild(option);
//     });
//
//     // Show/hide filter when "Node Performance" is selected
//     document.querySelectorAll("input[name='info-type']").forEach(radio => {
//         radio.addEventListener("change", function () {
//             metricsFilter.style.display = performanceRadio.checked ? "block" : "none";
//         });
//     });
// });
// document.addEventListener("click", function () {
// document.addEventListener("click", function () {
//     const performanceRadio = document.getElementById("performance-info-checkbox");
//     const metricsFilter = document.getElementById("metrics-filter");
//     const metricsSelect = document.getElementById("metrics-select");
//
//     // List of available metrics (update if needed)
//     const availableMetrics = ["Accuracy", "Auc", "Auprc", "BalancedAccuracy", "F1Score", "LogLoss", "MCC", "NPV", "PPV", "Precision", "Recall", "Sensitivity", "Specificity"];
//
//     // Populate dropdown with metric options
//     availableMetrics.forEach(metric => {
//         const option = document.createElement("option");
//         option.value = metric;
//         option.textContent = metric;
//         option.selected = true; // Select all by default
//         metricsSelect.appendChild(option);
//     });
//
//     // Store original `.node-content` values for each node **before any filtering happens**
//     document.querySelectorAll(".node-container").forEach(node => {
//         const contentDiv = node.querySelector(".node-content");
//         if (contentDiv) {
//             contentDiv.dataset.originalContent = contentDiv.innerHTML; // Save original full content
//         }
//     });
//
//     // Show/hide filter when "Node Performance" is selected
//     document.querySelectorAll("input[name='info-type']").forEach(radio => {
//         radio.addEventListener("change", function () {
//             if (performanceRadio.checked) {
//                 metricsFilter.style.display = "block";
//                 applyMetricFilter(); // Apply filter immediately when switching
//             } else {
//                 metricsFilter.style.display = "none";
//                 restoreAllMetrics(); // Restore all metrics when switching away
//             }
//         });
//     });
//
//     // Apply filtering when metric selection changes
//     metricsSelect.addEventListener("change", applyMetricFilter);
//
//     // function applyMetricFilter() {
//     //     const selectedMetrics = Array.from(metricsSelect.selectedOptions).map(option => option.value);
//     //
//     //     document.querySelectorAll(".node-container").forEach(node => {
//     //         const contentDiv = node.querySelector(".node-content");
//     //         if (contentDiv) {
//     //             // **Retrieve original content before filtering**
//     //             const originalContent = contentDiv.dataset.originalContent;
//     //             if (!originalContent) return;
//     //
//     //             // **Reset content first to prevent accumulating removals**
//     //             let filteredMetrics = originalContent.split("<br>").filter(metricLine => {
//     //                 return selectedMetrics.some(selectedMetric =>
//     //                     metricLine.trim().toLowerCase().startsWith(selectedMetric.toLowerCase() + ":")
//     //                 );
//     //             });
//     //
//     //             // **Ensure at least one metric remains to prevent empty display**
//     //             contentDiv.innerHTML = filteredMetrics.length > 0 ? filteredMetrics.join("<br>") : "(No metrics selected)";
//     //         }
//     //     });
//     // }
//
//     function applyMetricFilter() {
//         const selectedMetrics = Array.from(metricsSelect.selectedOptions).map(option => option.value.toLowerCase());
//
//         document.querySelectorAll(".node-container").forEach(node => {
//             node.querySelectorAll(".metric").forEach(metricElement => {
//                 const metricName = metricElement.dataset.metric.toLowerCase();
//                 metricElement.style.display = selectedMetrics.includes(metricName) ? "block" : "none";
//             });
//         });
//     }
//
//
//     function restoreAllMetrics() {
//         document.querySelectorAll(".node-container").forEach(node => {
//             const contentDiv = node.querySelector(".node-content");
//             if (contentDiv && contentDiv.dataset.originalContent) {
//                 contentDiv.innerHTML = contentDiv.dataset.originalContent; // Restore full original content
//             }
//         });
//     }
// });
