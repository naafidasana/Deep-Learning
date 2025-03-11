// Search and filtering functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('projectSearch');
    const stageFilter = document.getElementById('stageFilter');
    const algorithmFilter = document.getElementById('algorithmFilter');
    const projectCards = document.querySelectorAll('.project-card');
    
    function filterProjects() {
        const searchTerm = searchInput.value.toLowerCase();
        const stageValue = stageFilter.value;
        const algorithmValue = algorithmFilter.value;
        
        projectCards.forEach(card => {
            const title = card.querySelector('h4').textContent.toLowerCase();
            const content = card.querySelector('.project-content p').textContent.toLowerCase();
            const tags = Array.from(card.querySelectorAll('.tag')).map(tag => tag.textContent.toLowerCase());
            const cardStage = card.dataset.stage;
            const cardAlgorithm = card.dataset.algorithm;
            
            const matchesSearch = title.includes(searchTerm) || 
                                    content.includes(searchTerm) || 
                                    tags.some(tag => tag.includes(searchTerm));
            const matchesStage = stageValue === 'all' || cardStage === stageValue;
            const matchesAlgorithm = algorithmValue === 'all' || cardAlgorithm === algorithmValue;
            
            if (matchesSearch && matchesStage && matchesAlgorithm) {
                card.classList.remove('hidden');
            } else {
                card.classList.add('hidden');
            }
        });
        
        // Check if any stage headers should be hidden (no visible projects)
        document.querySelectorAll('.stage-header').forEach(header => {
            const stageId = header.id;
            const hasVisibleProjects = Array.from(projectCards).some(card => 
                card.dataset.stage === stageId && !card.classList.contains('hidden')
            );
            
            if (stageValue === 'all') {
                if (hasVisibleProjects) {
                    header.classList.remove('hidden');
                } else {
                    header.classList.add('hidden');
                }
            } else {
                if (stageId === stageValue) {
                    header.classList.remove('hidden');
                } else {
                    header.classList.add('hidden');
                }
            }
            
            // Also check if the following note should be hidden
            const nextNote = header.nextElementSibling;
            if (nextNote && nextNote.classList.contains('note')) {
                if (header.classList.contains('hidden')) {
                    nextNote.classList.add('hidden');
                } else {
                    nextNote.classList.remove('hidden');
                }
            }
        });
    }
    
    searchInput.addEventListener('input', filterProjects);
    stageFilter.addEventListener('change', filterProjects);
    algorithmFilter.addEventListener('change', filterProjects);
});