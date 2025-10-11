import {useCallback} from 'react';
import useVideo from '@/common/components/video/editor/useVideo';
import {useAtomValue} from 'jotai';
import {litLoRAModeEnabledAtom} from '@/demo/atoms';
import Logger from '@/common/logger/Logger';

export default function GenerateLoRACandidatesButton() {
  const video = useVideo();
  const isLITLoRAModeEnabled = useAtomValue(litLoRAModeEnabledAtom);

  const handleGenerateCandidates = useCallback(async () => {
    if (!video || !isLITLoRAModeEnabled) {
      return;
    }

    try {
      Logger.info('Generating LoRA candidates...');
      await video.generateLoraCandidates();
    } catch (error) {
      Logger.error('Failed to generate LoRA candidates:', error);
    }
  }, [video, isLITLoRAModeEnabled]);

  if (!isLITLoRAModeEnabled) {
    return null;
  }

  return (
    <button
      onClick={handleGenerateCandidates}
      style={{
        padding: '10px 14px',
        backgroundColor: '#4ECDC4',
        color: 'white',
        border: 'none',
        borderRadius: '6px',
        cursor: 'pointer',
        fontSize: '13px',
        fontWeight: '600',
        margin: '12px',
        transition: 'transform 0.15s, box-shadow 0.15s',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translateY(-1px)';
        e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = 'none';
      }}>
      <div
        style={{
          width: '20px',
          height: '20px',
          borderRadius: '50%',
          backgroundColor: 'rgba(255,255,255,0.25)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '12px',
          fontWeight: 'bold',
        }}>
        ✨
      </div>
      Generate LoRA Candidates
    </button>
  );
}
