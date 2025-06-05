class SportsClassifier {
  constructor() {
    this.selectedImage = null
    this.isClassifying = false
    this.initializeElements()
    this.bindEvents()
    this.initializeNavigation()
  }

  initializeElements() {
    // Upload elements
    this.uploadArea = document.getElementById("uploadArea")
    this.uploadContent = document.getElementById("uploadContent")
    this.previewContent = document.getElementById("previewContent")
    this.fileInput = document.getElementById("fileInput")
    this.selectButton = document.getElementById("selectButton")
    this.changeButton = document.getElementById("changeButton")
    this.previewImage = document.getElementById("previewImage")
    this.fileName = document.getElementById("fileName")

    // Classify elements
    this.classifySection = document.getElementById("classifySection")
    this.classifyButton = document.getElementById("classifyButton")
    this.buttonText = document.getElementById("buttonText")

    // Error elements
    this.errorAlert = document.getElementById("errorAlert")
    this.errorMessage = document.getElementById("errorMessage")

    // Results elements
    this.resultsSection = document.getElementById("resultsSection")
    this.topClass = document.getElementById("topClass")
    this.topConfidence = document.getElementById("topConfidence")
    this.topProbability = document.getElementById("topProbability")
    this.topProgress = document.getElementById("topProgress")
    this.otherResults = document.getElementById("otherResults")
    this.totalClasses = document.getElementById("totalClasses")
    this.maxConfidence = document.getElementById("maxConfidence")
    this.mainClass = document.getElementById("mainClass")
  }

  bindEvents() {
    // File input events
    this.selectButton.addEventListener("click", () => this.fileInput.click())
    this.changeButton.addEventListener("click", () => this.fileInput.click())
    this.fileInput.addEventListener("change", (e) => this.handleFileSelect(e))

    // Drag and drop events
    this.uploadArea.addEventListener("click", () => {
      if (!this.selectedImage) {
        this.fileInput.click()
      }
    })

    this.uploadArea.addEventListener("dragover", (e) => this.handleDragOver(e))
    this.uploadArea.addEventListener("dragleave", (e) => this.handleDragLeave(e))
    this.uploadArea.addEventListener("drop", (e) => this.handleDrop(e))

    // Classify button
    this.classifyButton.addEventListener("click", () => this.classifyImage())
  }

  initializeNavigation() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      });
    });

    // Add active class to navigation links on scroll
    window.addEventListener('scroll', () => {
      let current = '';
      const sections = document.querySelectorAll('section[id]');
      
      sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.clientHeight;
        if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
          current = section.getAttribute('id');
        }
      });

      document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
          link.classList.add('active');
        }
      });
    });
  }

  handleFileSelect(event) {
    const files = event.target.files
    if (files && files.length > 0) {
      this.processFile(files[0])
    }
  }

  handleDragOver(event) {
    event.preventDefault()
    this.uploadArea.classList.add("drag-over")
  }

  handleDragLeave(event) {
    event.preventDefault()
    this.uploadArea.classList.remove("drag-over")
  }

  handleDrop(event) {
    event.preventDefault()
    this.uploadArea.classList.remove("drag-over")

    const files = event.dataTransfer.files
    if (files && files.length > 0) {
      this.processFile(files[0])
    }
  }

  processFile(file) {
    // Validate file type
    if (!file.type.startsWith("image/")) {
      this.showError("Por favor, selecione um arquivo de imagem válido.")
      return
    }

    // Validate file size (10MB limit)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
      this.showError("O arquivo é muito grande. Por favor, selecione uma imagem menor que 10MB.")
      return
    }

    this.selectedImage = file
    this.hideError()
    this.hideResults()
    this.showImagePreview(file)
    this.showClassifyButton()
  }

  showImagePreview(file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      this.previewImage.src = e.target.result
      this.fileName.textContent = file.name

      this.uploadContent.classList.add("hidden")
      this.previewContent.classList.remove("hidden")
    }
    reader.readAsDataURL(file)
  }

  showClassifyButton() {
    this.classifySection.classList.remove("hidden")
  }

  hideClassifyButton() {
    this.classifySection.classList.add("hidden")
  }

  showError(message) {
    this.errorMessage.textContent = message
    this.errorAlert.classList.remove("hidden")
    
    // Auto-hide error after 5 seconds
    setTimeout(() => {
      this.hideError()
    }, 5000)
  }

  hideError() {
    this.errorAlert.classList.add("hidden")
  }

  showResults() {
    this.resultsSection.classList.remove("hidden")
  }

  hideResults() {
    this.resultsSection.classList.add("hidden")
  }

  async classifyImage() {
    if (!this.selectedImage || this.isClassifying) return

    this.isClassifying = true
    this.hideError()
    this.updateClassifyButton(true)

    try {
      const formData = new FormData()
      formData.append("file", this.selectedImage)

      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Erro na classificação da imagem")
      }

      const data = await response.json()
      this.displayResults(data)
    } catch (error) {
      this.showError("Erro ao classificar a imagem. Verifique se a API está rodando.")
      console.error("Classification error:", error)
    } finally {
      this.isClassifying = false
      this.updateClassifyButton(false)
    }
  }

  async simulateClassification() {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 2000))

    // Generate mock results
    const sports = [
      "Futebol", "Basquete", "Tênis", "Vôlei", "Natação", 
      "Atletismo", "Ciclismo", "Boxe", "Judô", "Ginástica"
    ]
    
    const shuffled = [...sports].sort(() => 0.5 - Math.random())
    const topSport = shuffled[0]
    const topProbability = 0.7 + Math.random() * 0.25 // 70-95%
    
    const mockData = {
      top_class: {
        class_name: topSport,
        probability: topProbability
      },
      top3_classes: shuffled.slice(0, 3).map(sport => ({ class_name: sport })),
      top3_probabilities: [
        topProbability,
        Math.random() * (1 - topProbability) * 0.8,
        Math.random() * (1 - topProbability) * 0.4
      ]
    }

    this.displayResults(mockData)
  }

  updateClassifyButton(isLoading) {
    if (isLoading) {
      this.classifyButton.disabled = true
      this.buttonText.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Classificando...'
    } else {
      this.classifyButton.disabled = false
      this.buttonText.innerHTML = '<i class="fas fa-bolt"></i> Classificar Imagem'
    }
  }

  displayResults(data) {
    // Update top result
    this.topClass.textContent = data.top_class.class_name;
    this.topConfidence.textContent = `Confiança: ${this.formatProbability(data.top_class.probability)}`;
    this.topProbability.textContent = this.formatProbability(data.top_class.probability);

    // Animate progress bar
    setTimeout(() => {
      this.topProgress.style.width = `${data.top_class.probability * 100}%`;
    }, 100);

    // Update other results
    this.otherResults.innerHTML = "";
    data.top3_classes.forEach((classInfo, index) => {
      if (index > 0) { // Skip the first one as it's already shown as top result
        const probability = data.top3_probabilities[index];
        const resultElement = this.createOtherResult(classInfo.class_name, probability);
        this.otherResults.appendChild(resultElement);
      }
    });

    // Update stats
    this.totalClasses.textContent = data.top3_classes.length;
    this.maxConfidence.textContent = this.formatProbability(data.top_class.probability);
    this.mainClass.textContent = data.top_class.class_name;

    this.showResults();
  }

  createOtherResult(className, probability) {
    const resultDiv = document.createElement('div');
    resultDiv.className = 'result-item';
    
    const classSpan = document.createElement('span');
    classSpan.className = 'class-name';
    classSpan.textContent = className;
    
    const probSpan = document.createElement('span');
    probSpan.className = 'probability';
    probSpan.textContent = this.formatProbability(probability);
    
    const progressDiv = document.createElement('div');
    progressDiv.className = 'progress-bar';
    
    const progressInner = document.createElement('div');
    progressInner.className = 'progress';
    
    // Animate progress bar
    setTimeout(() => {
      progressInner.style.width = `${probability * 100}%`;
    }, 200);
    
    progressDiv.appendChild(progressInner);
    resultDiv.appendChild(classSpan);
    resultDiv.appendChild(probSpan);
    resultDiv.appendChild(progressDiv);
    
    return resultDiv;
  }

  formatProbability(probability) {
    return `${(probability * 100).toFixed(1)}%`
  }
}

// Initialize the application when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new SportsClassifier()
})