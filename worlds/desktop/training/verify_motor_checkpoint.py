from pathlib import Path
import argparse, torch
from worlds.desktop.neural import ACTION_KINDS, NeuralDesktopBody, NeuralDesktopConfig

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,default=Path('state/neural_desktop.pt')); a=ap.parse_args()
 p=torch.load(a.checkpoint,map_location='cpu',weights_only=False); m=NeuralDesktopBody(NeuralDesktopConfig(**p['config'])); m.load_state_dict(p['model'],strict=False); m.eval()
 print({'motor_repaired':p.get('motor_repaired'),'motor_repair_version':p.get('motor_repair_version')})
 for name,x in [('black',torch.zeros(1,3,m.config.image_height,m.config.image_width)),('white',torch.ones(1,3,m.config.image_height,m.config.image_width)),('noise',torch.rand(1,3,m.config.image_height,m.config.image_width))]:
  with torch.no_grad(): o=m.propose(m.observe(x)); q=o.action_logits.softmax(-1)[0]
  print(name,ACTION_KINDS[int(q.argmax())],{k:round(float(v),4) for k,v in zip(ACTION_KINDS,q)})
if __name__=='__main__': main()
