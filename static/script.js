const startBtn = document.querySelector("#startBtn");
const navStartBtn = document.querySelector("#navStartBtn");
const wizard = document.querySelector("#wizard");

const itemForm = document.querySelector("#itemForm");
const steps = Array.from(document.querySelectorAll(".step"));
const currentStepText = document.querySelector("#currentStepText");
const stepTitle = document.querySelector("#stepTitle");
const progressFill = document.querySelector("#progressFill");

const prevBtn = document.querySelector("#prevBtn");
const nextBtn = document.querySelector("#nextBtn");
const submitBtn = document.querySelector("#submitBtn");

const summaryStep2 = document.querySelector("#summaryStep2");
const summaryStep3 = document.querySelector("#summaryStep3");
const summaryStep4 = document.querySelector("#summaryStep4");

const itemNameInput = document.querySelector("#itemName");
const brandInput = document.querySelector("#brand");
const summaryName = document.querySelector("#summaryName");
const summaryBrand = document.querySelector("#summaryBrand");

const usagePeriodInput = document.querySelector("#usagePeriod");
const usageFrequencyInput = document.querySelector("#usageFrequency");

const itemImageInput = document.querySelector("#itemImage");
const imagePreview = document.querySelector("#imagePreview");
const uploadPlaceholder = document.querySelector("#uploadPlaceholder");

const emotionalDescription = document.querySelector("#emotionalDescription");
const emotionDescCount = document.querySelector("#emotionDescCount");

const analysisCard = document.querySelector("#analysisCard");
const resultCard = document.querySelector("#resultCard");

let currentStep = 1;
let uploadedImageDataUrl = "";

const stepTitles = {
  1: "今天想整理哪一類物品？",
  2: "先認識一下這個物品",
  3: "它現在還常出現在你的生活裡嗎？",
  4: "最後，描述它的狀態與意義"
};

const categoryNames = {
  furniture_bedding: "家具寢具",
  electronics_3c: "家電與3C",
  clothing_accessories: "服飾配件",
  beauty_personal_care: "美妝個護",
  books_office: "書籍辦公",
  kitchen_living: "生活廚具",
  sports_hobbies: "運動愛好",
  memorabilia: "紀念品",
  other: "其他"
};

const usagePeriodNames = {
  within_1_year: "1 年內",
  "1_to_3_years": "1-3 年",
  "3_to_5_years": "3-5 年",
  "5_to_8_years": "5-8 年",
  over_8_years: "8 年以上"
};

const usageFrequencyNames = {
  new: "全新未使用",
  daily: "每天使用",
  weekly: "一週使用",
  monthly: "每月使用",
  yearly: "每年使用"
};

function scrollToWizard() {
  wizard.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}

startBtn.addEventListener("click", scrollToWizard);
navStartBtn.addEventListener("click", scrollToWizard);

prevBtn.addEventListener("click", () => {
  if (currentStep > 1) {
    currentStep--;
    updateStep();
  }
});

nextBtn.addEventListener("click", () => {
  if (!validateCurrentStep()) return;

  if (currentStep < 4) {
    currentStep++;
    updateStep();
  }
});

itemNameInput.addEventListener("input", updateSummary);
brandInput.addEventListener("input", updateSummary);

usagePeriodInput.addEventListener("change", updateChoiceSummaries);
usageFrequencyInput.addEventListener("change", updateChoiceSummaries);

function updateSummary() {
  const itemName = itemNameInput.value.trim();
  const brand = brandInput.value.trim();

  summaryName.textContent = itemName || "尚未命名的物品";
  summaryBrand.textContent = brand ? `品牌：${brand}` : "品牌未填寫";

  updateChoiceSummaries();
}

itemImageInput.addEventListener("change", () => {
  const file = itemImageInput.files[0];

  if (!file) return;

  const reader = new FileReader();

  reader.onload = function (event) {
    uploadedImageDataUrl = event.target.result;
    imagePreview.src = uploadedImageDataUrl;
    imagePreview.style.display = "block";
    uploadPlaceholder.style.display = "none";
  };

  reader.readAsDataURL(file);
});

emotionalDescription.addEventListener("input", () => {
  const maxLength = 128;
  const currentLength = emotionalDescription.value.length;
  emotionDescCount.textContent = maxLength - currentLength;
});

itemForm.addEventListener("submit", async function (event) {
  event.preventDefault();

  if (!validateCurrentStep()) return;

  const selectedCategory = document.querySelector('input[name="category"]:checked');

  const formData = {
    item_name: document.querySelector("#itemName").value,
    brand: document.querySelector("#brand").value,
    category: selectedCategory.value,
    usage_period: document.querySelector("#usagePeriod").value,
    usage_frequency: document.querySelector("#usageFrequency").value,
    objective_description: document.querySelector("#objectiveDescription").value,
    emotional_description: document.querySelector("#emotionalDescription").value
  };

  startAnalysisAnimation();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(formData)
    });


    if (!response.ok) throw new Error("後端錯誤");
    const data = await response.json();
    stopAnalysisAnimation();
    renderResult(data);

  } catch (error) {
    analysisCard.classList.add("hidden");
    resultCard.classList.remove("hidden");

    resultCard.innerHTML = `
      <div class="result-header">
        <h2>目前無法取得評估結果</h2>
        <span class="result-badge">錯誤</span>
      </div>

      <p class="reason">
        請確認 FastAPI 後端是否正在執行，或檢查前後端欄位名稱是否一致。
      </p>
    `;

    resultCard.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });
  }
});

function updateStep() {
  steps.forEach((step) => {
    const stepNumber = Number(step.dataset.step);
    step.classList.toggle("active", stepNumber === currentStep);
  });

  currentStepText.textContent = currentStep;
  stepTitle.textContent = stepTitles[currentStep];
  progressFill.style.width = `${(currentStep / 4) * 100}%`;

  updateChoiceSummaries();

  prevBtn.style.display = currentStep === 1 ? "none" : "inline-block";
  nextBtn.style.display = currentStep === 4 ? "none" : "inline-block";
  submitBtn.style.display = currentStep === 4 ? "inline-block" : "none";

  wizard.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}

function updateChoiceSummaries() {
  const selectedCategory = document.querySelector('input[name="category"]:checked');
  const categoryText = selectedCategory
    ? categoryNames[selectedCategory.value]
    : "尚未選擇";

  const itemName = itemNameInput.value.trim() || "尚未輸入";
  const brand = brandInput.value.trim() || "未填寫";

  const usagePeriodValue = usagePeriodInput.value;
  const usageFrequencyValue = usageFrequencyInput.value;

  const usagePeriodText = usagePeriodValue
    ? usagePeriodNames[usagePeriodValue]
    : "尚未選擇";

  const usageFrequencyText = usageFrequencyValue
    ? usageFrequencyNames[usageFrequencyValue]
    : "尚未選擇";

  if (summaryStep2) {
    summaryStep2.innerHTML = `
      <span>Selected category</span>
      <strong>${categoryText}</strong>
    `;
  }

  if (summaryStep3) {
    summaryStep3.innerHTML = `
      <span>Selected category</span>
      <strong>${categoryText}</strong>

      <span>Item</span>
      <strong>${itemName}</strong>

      <span>Brand</span>
      <strong>${brand}</strong>
    `;
  }

  if (summaryStep4) {
    summaryStep4.innerHTML = `
      <span>Selected category</span>
      <strong>${categoryText}</strong>

      <span>Item</span>
      <strong>${itemName}</strong>

      <span>Brand</span>
      <strong>${brand}</strong>

      <span>Used for</span>
      <strong>${usagePeriodText}</strong>

      <span>Frequency</span>
      <strong>${usageFrequencyText}</strong>
    `;
  }
}

function validateCurrentStep() {
  if (currentStep === 1) {
    const selectedCategory = document.querySelector('input[name="category"]:checked');

    if (!selectedCategory) {
      alert("請先選擇一個物品類別");
      return false;
    }
  }

  if (currentStep === 2) {
    const itemName = document.querySelector("#itemName").value.trim();

    if (!itemName) {
      alert("請輸入物品名稱");
      return false;
    }
  }

  if (currentStep === 3) {
    const usagePeriod = document.querySelector("#usagePeriod").value;
    const usageFrequency = document.querySelector("#usageFrequency").value;

    if (!usagePeriod || !usageFrequency) {
      alert("請選擇使用時間與使用頻率");
      return false;
    }
  }

  return true;
}

let animationInterval;

function startAnalysisAnimation() {
  resultCard.classList.add("hidden");
  analysisCard.classList.remove("hidden");

  const steps = document.querySelectorAll(".analysis-step");
  let index = 0;

  steps.forEach(s => s.classList.remove("active"));

  analysisCard.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });

  animationInterval = setInterval(() => {
    steps.forEach(s => s.classList.remove("active"));

    steps[index].classList.add("active");

    index = (index + 1) % steps.length; // 循環
  }, 450);
}


function stopAnalysisAnimation() {
  clearInterval(animationInterval);

  const steps = document.querySelectorAll(".analysis-step");
  steps.forEach(s => s.classList.remove("active"));
}


function showAnalysis() {
  resultCard.classList.add("hidden");
  analysisCard.classList.remove("hidden");

  const analysisSteps = Array.from(document.querySelectorAll(".analysis-step"));

  analysisSteps.forEach((step) => {
    step.classList.remove("active");
  });

  analysisCard.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });

  analysisSteps.forEach((step, index) => {
    setTimeout(() => {
      analysisSteps.forEach((item) => item.classList.remove("active"));
      step.classList.add("active");
    }, index * 450);
  });
}

function renderResult(data) {
  analysisCard.classList.add("hidden");
  resultCard.classList.remove("hidden");

  const imageBlock = uploadedImageDataUrl
    ? `<img src="${uploadedImageDataUrl}" alt="物品照片" />`
    : `<span>未上傳物品照片</span>`;

  // 格式化美金顯示
  const priceDisplay = data.secondhand_price_usd !== undefined 
    ? `預估 $${data.secondhand_price_usd}` 
    : "";

  resultCard.innerHTML = `
    <div class="result-header">
      <h2>${data.item_name || "這個物品"} 的整理建議</h2>
      <span class="result-badge">${data.recommendation}</span>
    </div>

    <div class="result-grid">
      <div class="result-photo">
        ${imageBlock}
      </div>

      <div class="score-grid">
        ${createScoreItem("總價值分數", data.total_score)}
        ${createScoreItem("使用價值", data.use_value)}
        ${createScoreItem("情感價值", data.emotional_value)}
        ${createScoreItem("二手價值", data.secondhand_value, priceDisplay)}
      </div>
    </div>

    <p class="reason">
      <strong>原因說明：</strong><br>
      ${data.reason}
    </p>
  `;

  resultCard.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}

function createScoreItem(title, score, extraText = "") {
  const extraHtml = extraText ? `<span class="score-extra">${extraText}</span>` : "";
  return `
    <div class="score-item">
      <div class="score-title">
        <div>
          <span>${title}</span>
          ${extraHtml}
        </div>
        <span>${score} / 100</span>
      </div>
      <div class="bar">
        <div class="bar-fill" style="width: ${score}%;"></div>
      </div>
    </div>
  `;
}

updateStep();
updateSummary();
updateChoiceSummaries();