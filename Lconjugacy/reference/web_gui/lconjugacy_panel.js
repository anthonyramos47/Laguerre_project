/**
 * lconjugacy_panel.js — Action buttons for the L-Conjugacy direction field GUI.
 *
 * All sliders/checkboxes are built automatically by AutoPanel from the
 * @gui_state schema. This file only adds the action buttons grouped by panel.
 */

import { Panel } from 'kayviz/gui/panel.js';

export class LconjugacyPanel extends Panel {
  constructor(ws, schema) {
    super('L-Conjugacy', { width: 340 });
    this.ws = ws;

    // BSpline section
    const bs = this.addSection('B-Spline');
    bs.add({ fn: () => ws.send('resample_bspline',      {}) }, 'fn').name('Resample');
    bs.add({ fn: () => ws.send('compute_reference',     {}) }, 'fn').name('Compute Reference');
    bs.add({ fn: () => ws.send('visualize_reference',   {}) }, 'fn').name('Show Reference');
    bs.add({ fn: () => ws.send('load_pre_remesh_state', {}) }, 'fn').name('Load Pre-Remesh State');

    // Direction field section
    const df = this.addSection('Direction Field');
    df.add({ fn: () => ws.send('init_direction_field', { method: 'constant' }) }, 'fn')
      .name('Init (Constant)');
    df.add({ fn: () => ws.send('init_direction_field', { method: 'linear'   }) }, 'fn')
      .name('Init (Linear)');
    df.add({ fn: () => ws.send('init_direction_field', { method: 'linearU'  }) }, 'fn')
      .name('Init (Linear U)');
    df.add({ fn: () => ws.send('init_direction_field', { method: 'linearV'  }) }, 'fn')
      .name('Init (Linear V)');
    df.add({ fn: () => ws.send('setup_df_optimizer',   {}) }, 'fn').name('Setup Optimizer');
    df.add({ fn: () => ws.send('run_df_optimizer',     { max_its: 40 }) }, 'fn').name('Run');
    df.add({ fn: () => ws.send('stop_df_optimizer',    {}) }, 'fn').name('Stop');
    df.add({ fn: () => ws.send('visualize_cross_field', {}) }, 'fn').name('Show Cross Field');
    df.add({ fn: () => ws.send('save_cross_field',      {}) }, 'fn').name('Save Cross Field');

    // Remeshing section
    const rm = this.addSection('Remeshing');
    rm.add({ fn: () => ws.send('save_pre_remesh_state', {}) }, 'fn').name('Save Pre-Remesh State');
    rm.add({ fn: () => ws.send('run_remesher',  { integrated: true  }) }, 'fn').name('Run Remesher (Integration)');
    rm.add({ fn: () => ws.send('run_remesher',  { integrated: false }) }, 'fn').name('Run Remesher (Direct)');
    rm.add({ fn: () => ws.send('load_remeshed', {}) }, 'fn').name('Load Remeshed');
    rm.add({ fn: () => ws.send('init_lmesh',    {}) }, 'fn').name('Initialize L-mesh');

    // Post-optimization section
    const po = this.addSection('Post Optimization');
    po.add({ fn: () => ws.send('run_post_optimizer',  { max_its: 50 }) }, 'fn').name('Run');
    po.add({ fn: () => ws.send('stop_post_optimizer', {}) }, 'fn').name('Stop');
    po.add({ fn: () => ws.send('show_lmesh',          {}) }, 'fn').name('Show L-mesh');

    // Save section
    const sv = this.addSection('Save');
    sv.add({ fn: () => ws.send('save_all', {}) }, 'fn').name('Save All');
  }
}

export default LconjugacyPanel;
