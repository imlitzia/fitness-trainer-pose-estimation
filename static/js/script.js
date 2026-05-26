document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const exerciseGrid = document.getElementById('exercise-grid');
    const exerciseCount = document.getElementById('exercise-count');
    const exerciseInfo = document.getElementById('exercise-info');
    const infoName = document.getElementById('info-name');
    const infoType = document.getElementById('info-type');
    const infoDescription = document.getElementById('info-description');
    const infoNote = document.getElementById('info-note');
    const categoryTabs = document.querySelectorAll('.category-tab');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const setsInput = document.getElementById('sets');
    const repsInput = document.getElementById('reps');
    const currentExercise = document.getElementById('current-exercise');
    const currentSet = document.getElementById('current-set');
    const currentReps = document.getElementById('current-reps');
    const formScore = document.getElementById('form-score');
    const formGrade = document.getElementById('form-grade');
    const fatigueScore = document.getElementById('fatigue-score');
    const fatigueLevel = document.getElementById('fatigue-level');
    const fatigueSignals = document.getElementById('fatigue-signals');
    const fatigueMessages = document.getElementById('fatigue-messages');
    const sigVelocity = document.getElementById('sig-velocity');
    const sigRom = document.getElementById('sig-rom');
    const sigShakiness = document.getElementById('sig-shakiness');
    const sigPause = document.getElementById('sig-pause');
    
    // Camera elements
    const videoElement = document.getElementById('video');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const faceVideoElement = document.getElementById('face-video');
    const facePlaceholder = document.getElementById('face-placeholder');
    const startCameraBtn = document.getElementById('start-camera-btn');
    const facialFatigueScore = document.getElementById('facial-fatigue-score');
    const facialFatigueLevel = document.getElementById('facial-fatigue-level');
    const faceTracking = document.getElementById('face-tracking');
    const facialSignals = document.getElementById('facial-signals');
    const facialMessages = document.getElementById('facial-messages');
    const sigEar = document.getElementById('sig-ear');
    const sigMouth = document.getElementById('sig-mouth');
    const sigBlink = document.getElementById('sig-blink');
    const sigHead = document.getElementById('sig-head');
    
    // Camera state
    let cameraStarted = false;
    
    // Variables
    let selectedExercise = null;
    let exercisesData = {};
    let workoutRunning = false;
    let statusCheckInterval = null;
    let faceStatusInterval = null;
    let currentCategory = 'all';
    
    // ==================== Camera Control Functions ====================
    function showCameraError(message) {
        const text = message || 'Could not start camera.';
        if (videoPlaceholder) {
            videoPlaceholder.style.display = 'flex';
            const hint = videoPlaceholder.querySelector('.camera-error-hint');
            if (hint) {
                hint.textContent = text;
            } else {
                const p = document.createElement('p');
                p.className = 'camera-error-hint';
                p.style.color = '#ffb4b4';
                p.textContent = text;
                videoPlaceholder.querySelector('.placeholder-content')?.appendChild(p);
            }
        }
        if (facePlaceholder) {
            facePlaceholder.style.display = 'flex';
        }
        if (videoElement) {
            videoElement.style.display = 'none';
            videoElement.removeAttribute('src');
        }
        if (faceVideoElement) {
            faceVideoElement.style.display = 'none';
            faceVideoElement.removeAttribute('src');
        }
        cameraStarted = false;
    }

    function beginVideoStreams() {
        const ts = Date.now();
        videoElement.src = `/video_feed?ts=${ts}`;
        videoElement.style.display = 'block';
        if (videoPlaceholder) {
            videoPlaceholder.style.display = 'none';
        }
        setTimeout(() => {
            if (!cameraStarted || !faceVideoElement) return;
            faceVideoElement.src = `/face_feed?ts=${ts}`;
            faceVideoElement.style.display = 'block';
            if (facePlaceholder) {
                facePlaceholder.style.display = 'none';
            }
        }, 800);
    }

    async function startCamera() {
        if (cameraStarted) return;

        if (startCameraBtn) {
            startCameraBtn.disabled = true;
            startCameraBtn.textContent = 'Starting...';
        }

        try {
            const res = await fetch('/start_camera', { method: 'POST' });
            const data = await res.json();
            if (!data.success) {
                showCameraError(data.error || 'Camera failed to open.');
                return;
            }

            const errHint = videoPlaceholder?.querySelector('.camera-error-hint');
            if (errHint) errHint.remove();

            console.log('Starting camera streams...');
            cameraStarted = true;
            beginVideoStreams();

            if (!faceStatusInterval) {
                faceStatusInterval = setInterval(() => {
                    fetch('/get_status')
                        .then(r => r.json())
                        .then(statusData => {
                            if (statusData.facial_fatigue_score !== undefined || statusData.face_tracking !== undefined) {
                                updateFacialFatigueDisplay(statusData);
                            }
                        })
                        .catch(() => {});
                }, 600);
            }
        } catch (err) {
            console.error(err);
            showCameraError('Server not reachable. Is python app.py running?');
        } finally {
            if (startCameraBtn) {
                startCameraBtn.disabled = false;
                startCameraBtn.textContent = 'Start Cameras';
            }
        }
    }
    
    function stopCamera() {
        if (!cameraStarted) return;
        
        console.log('Stopping cameras...');
        videoElement.src = '';
        videoElement.style.display = 'none';
        if (videoPlaceholder) {
            videoPlaceholder.style.display = 'flex';
        }
        if (faceVideoElement) {
            faceVideoElement.src = '';
            faceVideoElement.style.display = 'none';
        }
        if (facePlaceholder) {
            facePlaceholder.style.display = 'flex';
        }
        cameraStarted = false;
        if (faceStatusInterval) {
            clearInterval(faceStatusInterval);
            faceStatusInterval = null;
        }
        resetFacialUI();
        
        // Notify server to release camera
        fetch('/stop_camera', { method: 'POST' }).catch(() => {});
    }
    
    // Start camera button click
    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', startCamera);
    }
    
    // Stop camera when navigating away (not on refresh — avoids release/init race)
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function() {
            stopCamera();
        });
    });

    if (videoElement) {
        videoElement.addEventListener('error', () => {
            if (cameraStarted) {
                showCameraError('Body camera stream failed. Retry Start Cameras.');
            }
        });
    }
    if (faceVideoElement) {
        faceVideoElement.addEventListener('error', () => {
            console.warn('Face stream error (body stream may still work)');
        });
    }
    
    // Exercise categories mapping
    const exerciseCategories = {
        upper: ['push_up', 'hammer_curl', 'bicep_curl', 'tricep_dip', 'shoulder_press', 'lateral_raise'],
        lower: ['squat', 'lunge', 'side_lunge', 'deadlift', 'glute_bridge', 'calf_raise', 'wall_sit', 'leg_raise'],
        cardio: ['mountain_climber', 'high_knees', 'jumping_jack', 'plank']
    };
    
    // Exercise type info
    const exerciseTypeInfo = {
        'bilateral': {
            label: '↔️ Bilateral',
            color: '#9b59b6',
            note: '💡 This exercise alternates between left and right sides. Each side counts separately (e.g., left curl = 1, right curl = 2).'
        },
        'duration': {
            label: '⏱️ Duration',
            color: '#e67e22',
            note: '💡 This is a timed hold exercise. The counter shows seconds held in correct position.'
        },
        'standard': {
            label: '🔄 Standard',
            color: '#3498db',
            note: '💡 Standard repetition exercise. Each complete movement cycle counts as 1 rep.'
        }
    };
    
    // Exercise display names and descriptions
    const exerciseDetails = {
        'squat': { name: 'Squat', desc: 'Lower body compound movement targeting quads and glutes', icon: '🦵' },
        'push_up': { name: 'Push Up', desc: 'Upper body pushing exercise for chest and triceps', icon: '💪' },
        'hammer_curl': { name: 'Hammer Curl', desc: 'Bicep curl with neutral grip, alternating arms', icon: '💪' },
        'bicep_curl': { name: 'Bicep Curl', desc: 'Classic bicep exercise alternating between arms', icon: '💪' },
        'tricep_dip': { name: 'Tricep Dip', desc: 'Chair/bench dips for tricep strength', icon: '💪' },
        'shoulder_press': { name: 'Shoulder Press', desc: 'Overhead pressing for shoulder development', icon: '💪' },
        'lateral_raise': { name: 'Lateral Raise', desc: 'Side raises for shoulder width', icon: '💪' },
        'lunge': { name: 'Lunge', desc: 'Forward lunge alternating between legs', icon: '🦵' },
        'side_lunge': { name: 'Side Lunge', desc: 'Lateral lunge for inner thigh and glutes', icon: '🦵' },
        'deadlift': { name: 'Deadlift', desc: 'Hip hinge movement for hamstrings and back', icon: '🦵' },
        'glute_bridge': { name: 'Glute Bridge', desc: 'Hip bridge for glute activation', icon: '🦵' },
        'calf_raise': { name: 'Calf Raise', desc: 'Standing raises for calf muscles', icon: '🦵' },
        'wall_sit': { name: 'Wall Sit', desc: 'Isometric hold against wall for quad endurance', icon: '🦵' },
        'leg_raise': { name: 'Leg Raise', desc: 'Lying leg raises for lower abs', icon: '🔥' },
        'plank': { name: 'Plank', desc: 'Core stabilization hold exercise', icon: '🧘' },
        'mountain_climber': { name: 'Mountain Climber', desc: 'Dynamic cardio with alternating knee drives', icon: '🔥' },
        'high_knees': { name: 'High Knees', desc: 'Running in place with high knee lifts', icon: '🔥' },
        'jumping_jack': { name: 'Jumping Jack', desc: 'Classic full body cardio movement', icon: '🔥' }
    };
    
    // Load exercises from API
    async function loadExercises() {
        try {
            const response = await fetch('/exercises');
            const data = await response.json();
            exercisesData = data;
            exerciseCount.textContent = `(${data.count} exercises)`;
            renderExercises(data.exercises);
        } catch (error) {
            console.error('Error loading exercises:', error);
            exerciseGrid.innerHTML = '<div class="error">Failed to load exercises</div>';
        }
    }
    
    // Render exercise grid
    function renderExercises(exercises) {
        exerciseGrid.innerHTML = '';
        
        let filteredExercises = exercises;
        if (currentCategory !== 'all') {
            filteredExercises = exercises.filter(ex => exerciseCategories[currentCategory]?.includes(ex));
        }
        
        filteredExercises.forEach(exercise => {
            const info = exercisesData.info[exercise] || {};
            const details = exerciseDetails[exercise] || { name: exercise.replace(/_/g, ' '), desc: '', icon: '🏋️' };
            const typeInfo = exerciseTypeInfo[info.type] || exerciseTypeInfo['standard'];
            
            const div = document.createElement('div');
            div.className = 'exercise-option';
            div.setAttribute('data-exercise', exercise);
            div.setAttribute('data-type', info.type || 'standard');
            
            div.innerHTML = `
                <div class="exercise-icon">${details.icon}</div>
                <h3>${details.name}</h3>
                <span class="exercise-type-badge" style="background-color: ${typeInfo.color}">${typeInfo.label}</span>
            `;
            
            div.addEventListener('click', () => selectExercise(exercise, info));
            exerciseGrid.appendChild(div);
        });
    }
    
    // Select exercise
    function selectExercise(exercise, info) {
        // Remove previous selection
        document.querySelectorAll('.exercise-option').forEach(opt => opt.classList.remove('selected'));
        
        // Add selection to clicked
        const selected = document.querySelector(`[data-exercise="${exercise}"]`);
        if (selected) selected.classList.add('selected');
        
        selectedExercise = exercise;
        
        // Show exercise info
        const details = exerciseDetails[exercise] || { name: exercise, desc: '' };
        const typeInfo = exerciseTypeInfo[info.type] || exerciseTypeInfo['standard'];
        
        infoName.textContent = details.name;
        infoType.textContent = typeInfo.label;
        infoType.style.backgroundColor = typeInfo.color;
        infoDescription.textContent = details.desc;
        infoNote.innerHTML = typeInfo.note;
        
        exerciseInfo.classList.remove('hidden');
    }
    
    // Category tab switching
    categoryTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            categoryTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            currentCategory = this.getAttribute('data-category');
            renderExercises(exercisesData.exercises || []);
        });
    });
    
    // Start workout
    startBtn.addEventListener('click', function() {
        if (startBtn.disabled) return;

        if (!selectedExercise) {
            alert('Please select an exercise first!');
            return;
        }
        
        const sets = parseInt(setsInput.value);
        const reps = parseInt(repsInput.value);
        
        if (isNaN(sets) || sets < 1 || isNaN(reps) || reps < 1) {
            alert('Please enter valid numbers for sets and repetitions.');
            return;
        }

        startBtn.disabled = true;
        
        // Start exercise before camera: /video_feed blocks the server until threaded requests can run
        fetch('/start_exercise', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                exercise_type: selectedExercise,
                sets: sets,
                reps: reps
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                workoutRunning = true;
                stopBtn.disabled = false;

                startCamera();
                
                // Update UI
                const details = exerciseDetails[selectedExercise] || { name: selectedExercise };
                currentExercise.textContent = details.name;
                currentSet.textContent = `1 / ${sets}`;
                currentReps.textContent = `0 / ${reps}`;
                formScore.textContent = '100';
                formGrade.textContent = 'A';
                formGrade.className = 'status-value form-grade grade-a';
                updateFatigueDisplay({ fatigue_score: 100, fatigue_level: 'warming_up' });
                
                // Start status polling
                statusCheckInterval = setInterval(checkStatus, 500);
            } else {
                startBtn.disabled = false;
                alert('Failed to start exercise: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            startBtn.disabled = false;
            alert('An error occurred while starting the exercise.');
        });
    });
    
    // Stop workout
    stopBtn.addEventListener('click', function() {
        fetch('/stop_exercise', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                resetWorkoutUI();
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
    
    // Function to check status
    function checkStatus() {
        fetch('/get_status')
        .then(response => response.json())
        .then(data => {
            if (!data.exercise_running && workoutRunning) {
                // Workout has ended
                resetWorkoutUI();
                return;
            }
            
            // Update status display
            currentSet.textContent = `${data.current_set} / ${data.total_sets}`;
            currentReps.textContent = `${data.current_reps} / ${data.rep_goal}`;
            
            // Update form score
            if (data.form_score !== undefined) {
                formScore.textContent = Math.round(data.avg_form_score || data.form_score);
                const grade = data.form_grade || getGrade(data.form_score);
                formGrade.textContent = grade;
                formGrade.className = `status-value form-grade grade-${grade.toLowerCase()}`;
            }

            if (data.fatigue_score !== undefined) {
                updateFatigueDisplay(data);
            }

            if (data.facial_fatigue_score !== undefined || data.face_detected !== undefined) {
                updateFacialFatigueDisplay(data);
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
        });
    }
    
    function updateFatigueDisplay(data) {
        const score = data.fatigue_score ?? 100;
        const level = (data.fatigue_level || 'fresh').replace(/_/g, ' ');
        fatigueScore.textContent = `${score}%`;
        fatigueLevel.textContent = level;
        fatigueScore.className = `status-value fatigue-score level-${data.fatigue_level || 'fresh'}`;
        fatigueLevel.className = `status-value fatigue-level level-${data.fatigue_level || 'fresh'}`;

        const signals = data.fatigue_signals || data.signals || {};
        if (signals.velocity) {
            fatigueSignals.classList.remove('hidden');
            sigVelocity.textContent = `${Math.round(signals.velocity.ratio * 100)}%`;
            sigRom.textContent = `${Math.round((signals.rom?.ratio || 1) * 100)}%`;
            sigShakiness.textContent = `${Math.round((signals.shakiness?.ratio || 1) * 100)}%`;
            sigPause.textContent = `${Math.round((signals.pause?.ratio || 1) * 100)}%`;
        } else if (signals.reps_needed) {
            fatigueSignals.classList.remove('hidden');
            sigVelocity.textContent = '…';
            sigRom.textContent = '…';
            sigShakiness.textContent = data.live_shakiness != null ? String(data.live_shakiness) : '…';
            sigPause.textContent = '…';
        }

        const messages = data.fatigue_messages || data.messages || [];
        if (messages.length) {
            fatigueMessages.classList.remove('hidden');
            fatigueMessages.innerHTML = messages
                .map(m => `<p class="fatigue-msg">${m}</p>`)
                .join('');
        } else {
            fatigueMessages.classList.add('hidden');
            fatigueMessages.innerHTML = '';
        }
    }

    function updateFacialFatigueDisplay(data) {
        const score = data.facial_fatigue_score ?? 100;
        const level = (data.facial_fatigue_level || 'fresh').replace(/_/g, ' ');
        facialFatigueScore.textContent = `${score}%`;
        facialFatigueLevel.textContent = level;
        facialFatigueScore.className = `status-value facial-fatigue-score level-${data.facial_fatigue_level || 'fresh'}`;
        facialFatigueLevel.className = `status-value facial-fatigue-level level-${data.facial_fatigue_level || 'fresh'}`;

        if (faceTracking) {
            if (data.face_tracking) {
                faceTracking.textContent = 'Locked';
                faceTracking.style.color = '#27ae60';
            } else if (data.face_detected === false) {
                faceTracking.textContent = 'Searching';
                faceTracking.style.color = '#e67e22';
            } else {
                faceTracking.textContent = '--';
                faceTracking.style.color = '';
            }
        }

        const signals = data.facial_signals || {};
        if (signals.ear) {
            facialSignals.classList.remove('hidden');
            sigEar.textContent = `${Math.round((signals.ear.ratio || 1) * 100)}%`;
            sigMouth.textContent = `${Math.round((signals.mouth?.ratio || 1) * 100)}%`;
            sigBlink.textContent = signals.blink_rate_per_min != null ? String(signals.blink_rate_per_min) : '--';
            sigHead.textContent = `${Math.round((signals.head_ratio || 1) * 100)}%`;
        }

        const messages = data.facial_messages || [];
        if (messages.length) {
            facialMessages.classList.remove('hidden');
            facialMessages.innerHTML = messages.map(m => `<p class="fatigue-msg facial-msg">${m}</p>`).join('');
        } else {
            facialMessages.classList.add('hidden');
            facialMessages.innerHTML = '';
        }
    }

    function resetFacialUI() {
        if (!facialFatigueScore) return;
        facialFatigueScore.textContent = '--';
        facialFatigueLevel.textContent = '--';
        if (faceTracking) faceTracking.textContent = '--';
        facialSignals.classList.add('hidden');
        facialMessages.classList.add('hidden');
        facialMessages.innerHTML = '';
    }

    // Get grade from score
    function getGrade(score) {
        if (score >= 90) return 'A';
        if (score >= 80) return 'B';
        if (score >= 70) return 'C';
        if (score >= 60) return 'D';
        return 'F';
    }
    
    // Reset UI after workout ends
    function resetWorkoutUI() {
        workoutRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
        
        currentExercise.textContent = 'None';
        currentSet.textContent = '0 / 0';
        currentReps.textContent = '0 / 0';
        formScore.textContent = '--';
        formGrade.textContent = '--';
        formGrade.className = 'status-value form-grade';
        fatigueScore.textContent = '--';
        fatigueLevel.textContent = '--';
        fatigueScore.className = 'status-value fatigue-score';
        fatigueLevel.className = 'status-value fatigue-level';
        fatigueSignals.classList.add('hidden');
        fatigueMessages.classList.add('hidden');
        fatigueMessages.innerHTML = '';
        resetFacialUI();
    }
    
    // Initialize
    loadExercises();
});
