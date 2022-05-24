window.ismir = window.ismir || {};

(function (Tone, ns) {
  const AUDIO_DYNAMIC_RANGE_DB = 33;

  class XfadePlayer {
    constructor(toggleEl, stopEl, transportEl, xfadeEl, volumeEl) {
      const that = this;

      // Playback state
      this.__transport = null;

      // UI
      this.toggleEl = toggleEl;
      this.stopEl = stopEl;
      this.transportEl = transportEl;

      // Audio
      this.playerA = new Tone.Player();
      this.playerB = new Tone.Player();
      this.xfade = new Tone.CrossFade(0.5);
      this.volume = new Tone.Volume(0);
      this.playerA.connect(this.xfade.a);
      this.playerB.connect(this.xfade.b);
      this.xfade.connect(this.volume);
      this.volume.toDestination();

      // UI callbacks
      function onXfadeInput() {
        const xfade = xfadeEl.value / xfadeEl.max;
        that.xfade.fade.rampTo(xfade);
      }

      function onVolumeInput() {
        const volumeRel = volumeEl.value / volumeEl.max;
        if (volumeRel <= 0) {
          that.volume.mute = true;
        } else {
          that.volume.mute = false;
          that.volume.volume.rampTo(
            (Math.min(volumeRel, 1) - 1) * AUDIO_DYNAMIC_RANGE_DB
          );
        }
      }

      toggleEl.addEventListener("click", function () {
        if (that.__transport === null) return;
        if (that.__transport.startedAt === null) {
          that.play();
        } else {
          that.pause();
        }
      });
      stopEl.addEventListener("click", function () {
        that.stop();
      });
      transportEl.addEventListener("input", function () {
        if (that.__transport === null) return;
        that.setTransportOffset(
          (transportEl.value / transportEl.max) * that.__transport.duration
        );
      });
      volumeEl.addEventListener("input", onVolumeInput);
      xfadeEl.addEventListener("input", onXfadeInput);

      // UI init
      onXfadeInput();
      onVolumeInput();
      this.setLoaded(false);
    }

    getTransportOffset() {
      if (this.__transport === null) return 0;
      let offset = this.__transport.lastKnownOffset;
      if (this.__transport.startedAt !== null) {
        offset += Tone.now() - this.__transport.startedAt;
      }
      return offset;
    }

    setTransportOffset(offset) {
      if (this.__transport === null) return;
      if (this.__transport.startedAt === null) {
        this.__transport.lastKnownOffset = offset;
      } else {
        this.pause();
        this.__transport.lastKnownOffset = offset;
        this.play();
      }
      this.__tick();
    }

    pause() {
      if (this.__transport === null) return;
      this.playerA.stop();
      this.playerB.stop();
      const offset = this.getTransportOffset();
      this.__transport.startedAt = null;
      this.__transport.lastKnownOffset = offset;
    }

    stop() {
      if (this.__transport === null) return;
      this.playerA.stop();
      this.playerB.stop();
      this.__transport.startedAt = null;
      this.__transport.lastKnownOffset = 0;
    }

    play() {
      if (this.__transport === null) return;
      if (Tone.context.state !== "running") Tone.context.resume();
      const now = Tone.now();
      this.playerA.start(now, this.__transport.lastKnownOffset);
      this.playerB.start(now, this.__transport.lastKnownOffset);
      this.__transport.startedAt = now;
      this.__tick();
    }

    setLoaded(loaded) {
      this.stop();
      if (loaded) {
        this.__transport = {
          startedAt: null,
          lastKnownOffset: 0,
          duration: Math.max(
            this.playerA.buffer.duration,
            this.playerB.buffer.duration
          ),
        };
        this.toggleEl.disabled = false;
        this.stopEl.disabled = false;
        this.transportEl.disabled = false;
      } else {
        this.toggleEl.disabled = true;
        this.stopEl.disabled = true;
        this.transportEl.disabled = true;
        this.__transport = null;
      }
    }

    async setSources(a, b) {
      this.setLoaded(false);
      let result = true;
      try {
        await Promise.all([this.playerA.load(a), this.playerB.load(b)]);
      } catch {
        result = false;
      }
      this.setLoaded(result);
      return result;
    }

    __tick() {
      // Compute progress
      let progress = 0;
      if (this.__transport !== null) {
        const offset = this.getTransportOffset();
        progress = offset / this.__transport.duration;
      }

      // Update UI
      this.transportEl.value = Math.min(
        progress * this.transportEl.max,
        this.transportEl.max
      );

      // Stop transport
      if (progress < 1) {
        const that = this;
        window.requestAnimationFrame(function () {
          that.__tick();
        });
      } else {
        this.stop();
      }
    }
  }

  ns.XfadePlayer = XfadePlayer;
})(window.Tone, window.ismir);
