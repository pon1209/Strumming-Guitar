import {BasicPitch,outputToNotesPoly,noteFramesToTime} from '@spotify/basic-pitch';
let model;
window.transcribeLocal=async (audio,progress)=>{
 const ctx=new OfflineAudioContext(1,Math.ceil(audio.duration*22050),22050);
 const src=ctx.createBufferSource();src.buffer=audio;src.connect(ctx.destination);src.start();
 const mono=await ctx.startRendering();
 model ||= new BasicPitch(new URL('/model/model.json',location.href).href);
 const frames=[],onsets=[];
 try {await model.evaluateModel(mono,(f,o)=>{for(const x of f)frames.push(x);for(const x of o)onsets.push(x)},progress);
 if(!frames.length||!onsets.length)throw Error('Modelo no produjo matrices de notas');
 return noteFramesToTime(outputToNotesPoly(frames,onsets,0.35,0.3,5));
 } catch(e){model=undefined;throw e;}
};
