const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const previewText = document.getElementById("previewText");
const sendBtn = document.getElementById("sendBtn");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    previewText.style.display = "none";
});

sendBtn.addEventListener("click", async () => {
    const file = imageInput.files[0];
    if (!file) {
        alert("Upload an image first");
        return;
    }

    sendBtn.innerText = "Processing...";
    sendBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("http://localhost:8000/segmentations", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const text = await res.text();
            throw new Error(`Backend error: ${text}`);
        }

        const data = await res.json();

        if (!data.mask || !data.overlay) {
            throw new Error("Invalid response format");
        }

        document.getElementById("result1").src =
            "data:image/png;base64," + data.mask;

        document.getElementById("result2").src =
            "data:image/png;base64," + data.overlay;

    } catch (err) {
        console.error("Frontend error:", err);
        alert(err.message);
    } finally {
        sendBtn.innerText = "Process Image";
        sendBtn.disabled = false;
    }
});