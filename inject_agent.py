# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:23:48 2017

@author: simon

Simulate injection of an agent through a vascular network
Uses flow solutions from Paul Sweeney!

"""

from pymira import spatialgraph, interstitium
import pymira.front as frontPKG
import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt

def parker(t,delay):
    
    # PARKER (average arterial input function from human population)
    import numpy as np
    try:
        a1 = 0.833
        a2 = 0.336
        t1 = 0.171
        t2 = 0.364
        sigma1 = 0.055 
        sigma2 = 0.134
        alpha = 1.064
        beta = 0.166
        s = 37.772
        tau = 0.482
        tMin = (t-delay) / 60.
        r1 = (a1 / (sigma1*np.sqrt(2.*np.pi))) * np.exp(-np.square(tMin-t1)/(2.*np.square(sigma1)))
        r2 = (a2 / (sigma2*np.sqrt(2.*np.pi))) * np.exp(-np.square(tMin-t2)/(2.*np.square(sigma2)))
        wo = alpha * np.exp(-beta*tMin) / (1. + np.exp(-s*(tMin-tau)))
        conc = r1 + r2 + wo
        
        conc[tMin<0] = 0.
        conc[conc<0] = 0.
        conc[~np.isfinite(conc)] = 0.
        
    except Exception,e:
        print(e)
        import pdb
        pdb.set_trace()
        
    return conc
    
def ca1(t,delay):
    
    import numpy as np
    
    t_half = 9.02 * 60. * 60. # s (http://clincancerres.aacrjournals.org/content/10/4/1446.figures-only)
    cmax = 16.4 # ug/ml
    mol_weight = 580.237e6 # ug/mol
    dose = 100. #mg/kg
    mouse_mass = 0.025 # kg (mouse mass)
    mouse_volume = 1.8 #ml (blood volume...)
    dose_mass = dose * mouse_mass * 1000. # ug
    #dose_moles = dose_mass / mol_weight # mol
    cmax_mol = dose_mass * mouse_volume * 1e6 / mol_weight # umol
    
    return cmax_mol*np.exp(-(t-delay)/t_half)
    
def impulse(t,delay):
    
    import numpy as np
    length = 1. #s
    pos = delay
    conc = np.zeros(t.shape[0])
    conc[t>=delay] = 1.
    conc[t>(delay+length)] = 0.
    return conc

class InjectAgent(object):
    
    def __init__(self):
        self.dt = 0.2 # 60. #for CA1
        self.nt = 1500
        self.max_time = self.dt*self.nt
        # For CA1---
#        self.nt = 2000
#        self.max_time = 90.*60.
#        self.dt = self.max_time  / float(self.nt)
#        self.nt = int(self.max_time / self.dt)
        #-----
        self.time = np.arange(0,self.max_time,self.dt)
        self.output_times = np.arange(0,self.max_time,self.dt)
        # To convert from (nL/min) to (um^3/s) use conversion below
        self.fluxConversion = 1e6/60.
        
        self.Q_limit = 1e-9
        
        self.nodeList = None
        self.graph = None
        
        self.interstitium = interstitium.Interstitium()
        
    def vertex_flow_ordering(self,node):

        node.inFlow = 0.
        node.outFlow = 0.
        node.flow = []
        node.flow_direction = []
        node.delta_pressure = []
        node.distance = 0.
        node.is_inlet = False
        node.is_outlet = False
        
        if node.nconn==0:
            return
        
        for edge in node.edges:        
            reverse = True
            if edge.at_start_node(node.index):
                reverse = False

            pressure = edge.get_scalar('Pressure',reverse=reverse)
            flow = edge.get_scalar('Flow',reverse=reverse)
            radius = edge.get_scalar('Radii',reverse=reverse)
            
            delta_pressure = (pressure[-1]-pressure[0])
            if delta_pressure>0.:
                # Pressure increasing along vessel, towards node - inflow
                flowDir = -1
                eFlow = flow[0] #-np.abs(flow[0])
                node.inFlow += np.abs(flow[0])
            elif delta_pressure<0.:
                # Pressure decreasing along vessel, away from node - outflow
                flowDir = 1
                eFlow = flow[0] #np.abs(flow[0])
                node.outFlow += np.abs(flow[0])
            elif delta_pressure==0.:
                flowDir = 0
                eFlow = flow[0] #np.abs(flow[0])
                
            velocity = np.asarray([(f*self.fluxConversion)/(np.square(r)*np.pi) for f,r in zip(flow,radius)])
            length = np.zeros(edge.npoints)
            pts = edge.coordinates
            for i in range(1,edge.npoints):
                length[i] = np.linalg.norm(pts[i]-pts[i-1]) # Length (um)
    
            delay = np.asarray([np.sum(l/np.abs(v)) if v!=0. else 1.e6 for l,v in zip(length,velocity)]) # seconds
                
            node.flow_direction.append(flowDir)
            node.flow.append(eFlow)
            node.delta_pressure.append(delta_pressure)
            
            edge.velocity = velocity
            edge.length = length
            edge.delay = delay
            edge.concentration = np.zeros([edge.npoints,len(self.time)])

        # Sort out flow direction in edges with no pressure drop            
        if 0 in node.flow_direction:
            # Find combination that minimises inflow/outflow difference
            inFlow = [node.flow[i] for i,dir in enumerate(node.flow_direction) if dir>0]
            outFlow = [node.flow[i] for i,dir in enumerate(node.flow_direction) if dir<0]
            noPD = [node.flow[i] for i,dir in enumerate(node.flow_direction) if dir==0]
            noPDInds = [i for i,dir in enumerate(node.flow_direction) if dir==0]
            mnState = None
            mn = 1e6
            #print('{} zero pressure drop edges to fix'.format(len(noPDInds)))
            from itertools import product
            for i,state in enumerate(product([-1,1], repeat=len(noPDInds))): 
                prod = np.sum(inFlow)-np.sum(outFlow)-np.sum([v*s for (v,s) in zip(noPD,state)])
                if prod<mn:
                    mnState = state
                    mn = prod
            
            for i,ind in enumerate(noPDInds):
                node.flow_direction[ind] = mnState[i]
            node.inFlow = np.sum([node.flow[i] for i,dir in enumerate(node.flow_direction) if dir>0])
            node.outFlow = np.sum([node.flow[i] for i,dir in enumerate(node.flow_direction) if dir<0])
            
        node.is_inlet = False
        #if (node.outFlow>0.) and (node.inFlow==0.):
        if node.nconn==1 and node.outFlow>0:
            node.is_inlet = True
            
#        if node.nconn==1:
        if node.inFlow==0:
            node.inFlow = node.outFlow
        elif node.outFlow==0:
            node.outFlow = node.inFlow
            
        node.Q = np.zeros(node.nconn,dtype='float') + 0.
        for i,edge in enumerate(node.edges):
            if node.flow_direction[i]>=0:
                if node.inFlow!=0.:
                    #edge.Q = node.flow[i] / node.inFlow
                    node.Q[i] = node.flow[i] / node.inFlow
                    if node.Q[i]>1.001 and 0 in node.flow_direction:
                        import pdb
                        pdb.set_trace()
                        #pass
                else:
                    node.Q[i] = 0.
                    #edge.Q = 0.
            else:
                #edge.Q = 0.
                node.Q[i] = 0.
        
    def auc(self,edges):
        
        for edge in edges:
            auc = np.sum(edge.concentration,axis=1)*self.dt
            edge.add_scalar('AUC',auc)
            
    def add_concentration(self,edges,time,conc_time=0.):
        
        idx = (np.abs(time-conc_time)).argmin()
        #conc = np.zeros([len(edges),edges[0].concentration.shape[0]])        
        for i,edge in enumerate(edges):
            concVal = edge.concentration[:,idx]
            edge.add_scalar('Concentration_{}'.format(conc_time),concVal)
            
    def add_distance(self,edges):
        for i,edge in enumerate(edges):
            if getattr(edge,'distance',None) is not None:
                edge.add_scalar('Distance',[l+edge.distance for l in edge.length])
            else:
                edge.add_scalar('Distance',[-1.]*edge.npoints)
            
    def plot_conc(self,conc):
        plt.plot(self.time, conc)
            
    def inflow_ratio(self,nodeList):
        
        for node in nodeList:
            if node.outFlow>0.:
                r = node.inFlow / node.outFlow
            else:
                r = 0.
            node.add_scalar('Flow ratio',r)
            
    def get_concentration(self,edges):
        
        conc = np.zeros([len(edges),edges[0].concentration.shape[0]])        
        for i,edge in enumerate(edges):
            conc[i,:] = edge.concentration
            
    def reconstruct_results(self, graph, output_directory=None,recon_interstitium=True,name=None, recon_vascular=True):

        self.graph = graph
        
        if name is not None:
            output_directory = os.path.join(output_directory,name)

        import dill as pickle
        
        if recon_vascular:
            try:
                nodeFile = os.path.join(output_directory,'nodeList.dill')
                if not os.path.isfile(nodeFile):
                    print 'Generating node list...'
                    self.nodeList = self.graph.node_list()
                else:
                    with open(nodeFile ,'rb') as fo:
                        self.nodeList = pickle.load(fo)
                  
                self.save_graph(output_directory=output_directory)
            except Exception,e:
                print e,nodeFile
                import pdb
                pdb.set_trace()
               
        if recon_interstitium:
            print('Reconstructing interstitial results...')
            import scipy
            intObj = interstitium.Interstitium()
            interDir = os.path.join(output_directory,'interstitium_calcs')
            #interDir = r'C:\Users\simon\Dropbox\160113_paul_simulation_results\LS147T\1\impulseinterstitium_calcs'
            nadded = 0
            init = False
            if os.path.isdir(interDir):
                files = os.listdir(interDir)
                for i,f in enumerate(files):
                    try:
                        curFile = os.path.join(interDir,f)
                        data = np.load(curFile)
                        curgrid = data['grid']
                        print('Reconstructing file {} of {}: {}. Max conc: {}'.format(i+1,len(files),f,np.max(curgrid)))
                        grid_dims = data['grid_dims']
                        embedDims = data['embedDims']
                        if not init:
                            grid = curgrid
                        else:
                            grid += curgrid
                        nadded += 1
                        init = True
                        pixdim = [data['dx'],data['dy'],data['dz'],data['dt']]
                        boundingBox = data['embedDims'].flatten()
                    except Exception,e:
                        print('Error loading {}: {}'.format(curFile,e))
                #ofile = os.path.join(output_directory,'interstitium.nii')
                sm = False
                if sm:
                    medwinsize = 5
                    print('Median filtering: {}'.format(medwinsize))
                    #import pdb
                    #pdb.set_trace()
                    for j in xrange(grid.shape[0]):
                        #print j
                        grid[j] = scipy.ndimage.filters.median_filter(grid[j],size=medwinsize)
                        
                #pixdim = [data['dx'],data['dy'],data['dz'],data['dt']]
                
                from pymira import mesh
                timePoints = self.output_times
                odir = os.path.join(output_directory, 'interstitial_concentration_recon')
                if not os.path.isdir(odir):
                    os.mkdir(odir)
                #boundingBox = data['embedDims'].flatten()
                #timePoints = np.linspace(np.min(self.output_times),np.max(self.output_times),num=500)
                tp_early = np.linspace(0,10,num=20)
                tp_late = np.linspace(11,100,num=10)
                timePoints = np.append(tp_early,tp_late)

                for ti,tp in enumerate(timePoints): 
                    cur = grid[ti,:,:,:]
                    if True:
                       medwinsize = 5
                       #print('Median filtering: {}'.format(medwinsize))
                       cur = scipy.ndimage.filters.median_filter(cur,size=medwinsize)
                    m = mesh.Mesh(data=cur,boundingBox=boundingBox)
                    ofile = os.path.join(odir,'interstitial_conc_t{}.am'.format(ti))
                    print('Writing (max conc {}) {}'.format(np.max(cur),ofile))
                    m.add_parameter('Time',np.asscalar(tp))
                    m.write(ofile)
                #intObj.save_grid(output_directory,grid=grid,pixdim=pixdim,format='amira')
            print('Interstitial concentration grid written to {}'.format(output_directory))
            
    def save_graph(self,output_directory=None,remove_zeros=False):
         
        if self.graph is None or self.nodeList is None:
            return
          
        print('Generating edge list...')
        edges = self.graph.edges_from_node_list(self.nodeList)
            
        # Reconstruct individual inlet results
        eDir = os.path.join(output_directory,'edge_calcs')
        #eDir = r'C:\Users\simon\Dropbox\160113_paul_simulation_results\LS147T\1\impulseedge_calcs'
        files = os.listdir(eDir)
        nfiles = len(files)
        print('Loading concentration calc results ({} files)...'.format(len(files)))
        concImported = False
        init = False
        for fi,f in enumerate(files):
            print('Reading file {} of {}: {}'.format(fi+1,nfiles,f))
            with open(os.path.join(eDir,f),'rb') as fo:
                curEdges,ind = pickle.load(fo)
                
            if not init:
                nt = curEdges[0].concentration.shape[1]
                ntSrc = edges[0].concentration.shape[1]
                if nt!=ntSrc:
                    print('Fixing concentration shape...')
                    for e in edges:
                        e.concentration = np.zeros([e.npoints,nt])
                init = True

            for curEdge in curEdges:
                concImported = True
                srcEdge = [e for e in edges if e.index==curEdge.index]
                if all([i==j for i,j in zip(srcEdge[0].concentration.shape, curEdge.concentration.shape)]):
                    srcEdge[0].concentration += curEdge.concentration
                else:
                    print 'Shapes incompatible! {} {}'.format(srcEdge[0].concentration.shape,curEdge.concentration.shape)

        # Calculate AUC
        #print('Calculating AUC...')
        #self.auc(edges)
        print('Adding concentration-time data to graph')
        if False: #old version - adds multiple scalar fileds to a signel graph file
            # Add concentration(t=1s) as a scalar field
            timePoints = self.output_times
            for tp in timePoints:
                self.add_concentration(edges,self.time,conc_time=tp)
            print('Calculating distance...')
            self.add_distance(edges)
        else:
        #if True: #new version - creates multiple graph files (one per timepoint). Import into Amira with load timeseries
            # Add concentration(t=1s) as a scalar field
            timePoints = self.output_times
            odir = os.path.join(output_directory,'concentration_recon')
            if not os.path.isdir(odir):
                os.mkdir(odir)
            #timePoints = np.linspace(np.min(self.output_times),np.max(self.output_times),num=500)
            tp_early = np.linspace(0,10,num=20)
            tp_late = np.linspace(11,100,num=10)
            timePoints = np.append(tp_early,tp_late)
            for ti,tp in enumerate(timePoints):
                mx = 0.
                for edge in edges:
                    curConc = np.clip(edge.concentration[:,ti],0.,1e100)
                    curMax = np.max(curConc)
                    if curMax>mx:
                        mx = curMax
                    edge.add_scalar('Concentration',curConc)
                new_graph = self.graph.node_list_to_graph(self.nodeList)
                ofile = os.path.join(odir,'concentration_t{}.am'.format(ti))
                print('Writing {}, max conc {}'.format(ofile,mx))
                new_graph.add_parameter('Time',np.asscalar(tp))
                new_graph.write(ofile)
        
        # Not working yet!
        if remove_zeros:
            # Remove edges with AUC=0
            print('Identifying zero-concentration edges')
            edge_to_delete = [e.index for e in edges if np.all(e.get_scalar('AUC')==0.)]
            print('Deleting {} EDGES'.format(len(edge_to_delete)))
            node_to_delete = np.zeros(len(self.nodeList),dtype='int')
            ndelcount = 0
            for nInd,n in enumerate(self.nodeList):
                [n.remove_edge([eInd]) for eInd,e in enumerate(n.edges) if e.index in edge_to_delete]
                #if n.nconn==0:
                #    node_to_delete[nInd] = 1
                #    ndelcount += 1
            #print('Deleting {} NODES'.format(ndelcount))
            #self.nodeList = [n for nInd,n in enumerate(self.nodeList) if node_to_delete[nInd]==1]
        
        if False:
            print('Creating new graph...')
            new_graph = self.graph.node_list_to_graph(self.nodeList)
            if output_directory is not None:
                ofile = os.path.join(output_directory,'ct_output.am')
                print('Writing graph to file ({})...'.format(ofile))
                new_graph.write(ofile)
                print('Writing complete')
                
                #ofile = os.path.join(output_directory,'')+'ct.p'            
                #print('Saving concentration data ({})...'.format(ofile))
                #conc = self.get_concentration(edges)
                #with open(ofile, 'wb') as handle:
                #    pickle.dump(ofile, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def inject(self, graph, output_directory=None, resume=False, parallel=True, name=None, largest_inflow=False):

        self.graph = graph   

        import dill as pickle
        
        if name is not None:
            output_directory = os.path.join(output_directory,name)
            if not os.path.isdir(output_directory):
                os.mkdir(output_directory)
        
        if resume:
            eDir = os.path.join(output_directory,'edge_calcs')
            files = os.listdir(eDir)
            nfiles = len(files)
            inletVisited = []
            prefix = 'edges_inlet'
            suffix = '.dill'
            for fi,f in enumerate(files):
                try:
                    ind = int((f.replace('edges_inlet','')).replace('.dill',''))
                    inletVisited.append(ind)
                except Exception,e:
                    print('Could not load {}'.format(f))
        else:
            inletVisited = []
        
        nodeFile = os.path.join(output_directory,'nodeList.dill')
        #if True:
        if not os.path.isfile(nodeFile):
            print 'Generating node list...'
            nodeList = graph.node_list()
            self.nodeList = nodeList
            print 'Calculating flow ordering...'
            for node in nodeList:
                self.vertex_flow_ordering(node)
                nconn = np.asarray([x.nconn for x in self.nodeList])
        
            sTerminal = np.where(nconn==1)[0]
            nTerminal = len(sTerminal)
            
            sInlet = [node.index for node in self.nodeList if node.is_inlet==True]# and node.flow_direction[0]>=0]
            nInlet = len(sInlet)
            total_inflow = np.abs(np.sum([self.nodeList[i].flow[0] for i in sInlet]))
            inletNodes = [self.nodeList[x] for x in sInlet]
            for inletNode in inletNodes:
                inletNode.inletQ = inletNode.flow[0] / total_inflow
                inletNode.inletDelay = 0.
                
            # Limit to only inlet with highest Q
            highestQinlet = False
            if highestQinlet:
                inletQ = [n.outFlow for n in inletNodes]
                mxQind = np.argmax(inletQ)
                inletNodes = [inletNodes[mxQind]]
            
            print('Pickling node list...')
            with open(nodeFile,'wb') as fo:
                pickle.dump(nodeList,fo)
        else:
            with open(nodeFile ,'rb') as fo:
                self.nodeList = pickle.load(fo)
            inletNodes = [n for n in self.nodeList if n.is_inlet]
           
        inletNodes = [n for n in inletNodes if n.index not in inletVisited]
        
        if largest_inflow:
            inletNodes = [inletNodes[np.argmax([n.inletQ for n in inletNodes])]]
                
        edgeFile = os.path.join(output_directory,'edgeList.dill')
        if True:
        #if not os.path.isfile(nodeFile):
            edges = graph.edges_from_node_list(self.nodeList)
        #    with open(edgeFile,'wb') as fo:
        #        pickle.dump(edges,fo)
        else:
            with open(edgeFile ,'rb') as fo:
                edges = pickle.load(fo)
        nedge = len(edges)
            
        graphFile = os.path.join(output_directory,'graph.dill')
                
        timeFile = os.path.join(output_directory,'time.dill')
        #if not os.path.isfile(timeFile):
        if True:
            with open(timeFile,'wb') as fo:
                pickle.dump(self.time,fo)

        import pathos.multiprocessing as multiprocessing
        #import multiprocessing
        ncpu = multiprocessing.cpu_count()
        p = multiprocessing.ProcessingPool(ncpu)
        
        #intr = interstitium.Interstitium()
        #intr.set_grid_dimensions(graph.get_data('EdgePointCoordinates'),self.time)
        
        #argList = [[nodeFile,n.index,ca1,timeFile,output_directory,nedge,intr.grid_dims,intr.embedDims] for n in inletNodes]
        concFunc = parker
        argList = [[nodeFile,n.index,concFunc,timeFile,output_directory,nedge,None,None] for n in inletNodes]

        if parallel:
            p.map(_worker_function,argList)
        else:
            for arg in argList:
                _worker_function(arg)

        self.save_graph(output_directory=output_directory)
        
def _worker_function(args):
    
    import pymira.front as frontPKG
    import numpy as np
    import dill as pickle

    from pymira import interstitium

    def scale_and_shift(conc,time,Q=1.,delay=0.):
        dt = time[1]-time[0]
        d = delay / dt
        import scipy
        cnew = Q*scipy.ndimage.interpolation.shift(conc,d)
        return cnew.clip(min=0.)
    
    Q_limit = 1e-9
    c_limit = 1e-9
    
    nodeListFile,inletNodeIndex,concFunc,timeFile,odir,nedge,grid_dims,embed_dims = args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7]
    
    with open(nodeListFile ,'rb') as fo:
        nodeList = pickle.load(fo)
    with open(timeFile ,'rb') as fo:
        time = pickle.load(fo)
    
    nnode = len(nodeList)
    
    inletNode = [n for n in nodeList if n.index==inletNodeIndex][0]
    print('Inlet index: {}'.format(inletNode.index))
    
    vedges = []
    visited = np.zeros(nnode,dtype='bool') # []
    visited_edges = np.zeros(nedge,dtype='bool')
    edgesOut = []
    curNode = inletNode
    
    leaky_vessels = True
    ignore_delay = False
    if leaky_vessels:
        intr = interstitium.Interstitium()
        nodeCoords = np.asarray([n.coords for n in nodeList])
        intr.set_grid_dimensions(nodeCoords,time,grid_dims=grid_dims,embed_dims=embed_dims)
        grid = np.zeros(intr.grid_dims,dtype='float')
    
    #print('Inlet node info: delay: {} Q:{}'.format(inletNode.inletDelay,inletNode.inletQ))
    front = frontPKG.Front([inletNode],delay=inletNode.inletDelay,Q=inletNode.inletQ,distance=inletNode.distance,conc=None)
    
    # START WALK...
    endloop = False
    count = 0
    nStepMax = 1e3

    while endloop is False:
        count += 1
        if count>=nStepMax:
            endloop = True
        
        if len(front.Q)>0:
            mnQ = np.min(front.Q[0:front.front_size])
            mxQ = np.max(front.Q[0:front.front_size])
            mxC = np.max(front.conc[0:front.front_size])
            print('Inlet: {}., iteration: {}, front size: {},maxQ: {}, c_v(max): {}, c_i(max) {}'.format(inletNodeIndex,count,front.front_size,mnQ,mxC,np.max(intr.grid)))

        if front.front_size>0 and endloop is False:              
            (current_nodes,delay,Q,distance,concIn) = front.get_current_front()

            for n,curNode in enumerate(current_nodes):
                #print('Q: {}'.format(curNode.Q))
                #print('dP: {}'.format(curNode.delta_pressure))
                
                if not visited[curNode.index]:
                    visited[curNode.index] = True
                
                res = [(nodeList[nodeIndex],curNode.edges[i],curNode.Q[i]) for i,nodeIndex in enumerate(curNode.connecting_node) if curNode.Q[i]>0.]
                    
                if len(res)>0:
                    
                    # Select qualifying branches
                    next_nodes = [r[0] for r in res if r[2]!=0.]
                    via_edges = [r[1] for r in res if r[2]!=0.]
                    edge_Q = [r[2] for r in res if r[2]!=0.]
                    
                    delay_from = []
                    Q_from = []
                    distance_from = []
                    conc_from = [] # np.empty(0,dtype='float')
                    for ve,via_edge in enumerate(via_edges):                        
                        if via_edge not in vedges:
                            vedges.append(via_edge)
                        if not visited_edges[via_edge.index]:
                            visited_edges[via_edge.index] = True
                            edgesOut.append(via_edge)
                            via_edge.distance = distance[n]
                        
                        if leaky_vessels:
                            try:
                                if concIn[0] is None:
                                    conc = Q[n]*edge_Q[ve]*concFunc(time,delay[n])
                                else:
                                    if len(concIn)==1:
                                        nxtConc = concIn[0]
                                    else:
                                        nxtConc = concIn[n]
                                    if not ignore_delay:
                                        conc = scale_and_shift(nxtConc,time,Q=Q[n]*edge_Q[ve],delay=delay[n])
                                    else:
                                        conc = nxtConc * Q[n]*edge_Q[ve]
                                    
                                edgeInd = np.zeros(via_edge.npoints,dtype='int')
                                c_v = np.repeat([conc],via_edge.npoints,axis=0)
                                #import pdb
                                #pdb.set_trace()
                                c_v_out = intr.interstitial_diffusion(via_edge.coordinates,edgeInd,c_v,time,set_grid_dims=False,progress=False,store_results=True)
                                grid += intr.grid
                                #c_v_out = np.repeat([c_v_out],via_edge.npoints,axis=0)
                                via_edge.concentration = c_v_out
                            except Exception,err:
                                print('Error, interstitium calc {}'.format(err))
                        else:
                            try:
                                conc = Q[n]*edge_Q[ve]*concFunc(time,delay[n])
                                via_edge.concentration = np.repeat([conc],via_edge.npoints,axis=0)
                            except Exception,err:
                                print('Error, conc calc {}'.format(err))
                                                
                        delay_from.append(np.sum(via_edge.delay)+delay[n])
                        Q_from.append(Q[n]*edge_Q[ve])
                        distance_from.append(np.sum(via_edge.length)+distance[n])
                        if len(conc_from)==0:
                            conc_from = via_edge.concentration[-1,:]
                        elif len(conc_from.shape)==1 and conc_from.shape[0]!=0:
                            conc_from = conc_from[np.newaxis]
                            conc_from = np.append(conc_from,via_edge.concentration[-1,:][np.newaxis],axis=0)
                        else:
                            conc_from = np.append(conc_from,via_edge.concentration[-1,:][np.newaxis],axis=0)
                        
                    # Eliminate nodes that have a Q lower than limit
                    inds = [i for i,q in enumerate(Q_from) if q>Q_limit]# and np.max(conc_from[i])>c_limit]
                    #import pdb
                    #pdb.set_trace()
                    if len(inds)>0:
                        if len(conc_from.shape)==1:
                            nxtConc = [conc_from]
                        elif len(conc_from.shape)==2:
                            nxtConc = conc_from[inds,:]
                        front.step_front([next_nodes[i] for i in inds],
                                         delay=[delay_from[i] for i in inds],
                                         Q=[Q_from[i] for i in inds],
                                         distance=[distance_from[i] for i in inds],
                                         conc=nxtConc)
                    else:
                        pass
                else:
                    pass
            maxConc = [np.max(e.concentration) for e in vedges]
            sind = np.where(maxConc<=0.)
            if sind[0].shape[0]>0:
                print('Shapes incompatable!')
                #import pdb
                #pdb.set_trace()
            front.complete_step()
        elif count>=nStepMax:
            endloop = True
            print('Exit step {}, inlet {} - front size greater than maximum!'.format(inletNodeIndex,front.nstep))
        else:
            endloop = True
            print('Exit step {}, inlet {} - front size 0'.format(inletNodeIndex,front.nstep))
            #break
            
    import os
    eDir = os.path.join(odir,'edge_calcs')
    if not os.path.exists(eDir):
        os.makedirs(eDir)
    ofile = os.path.join(eDir,'edges_inlet{}.dill'.format(inletNodeIndex))
    #import pdb
    #pdb.set_trace()
    with open(ofile,'wb') as fo:
        pickle.dump((edgesOut,inletNodeIndex),fo)
        
    if leaky_vessels:
        intDir = os.path.join(odir,'interstitium_calcs')
        if not os.path.exists(intDir):
            os.makedirs(intDir)
        ofile = os.path.join(intDir ,'interstitium_inlet{}.npz'.format(inletNodeIndex))
        print('Grid min/max: {} {}'.format(np.min(grid),np.max(grid)))
        np.savez(ofile,grid=grid,grid_dims=intr.grid_dims,embedDims=intr.embedDims,dx=intr.dx,dy=intr.dy,dz=intr.dz,dt=intr.dt)
        #with open(ofile,'wb') as fo:
        #    pickle.dump((grid,inletNodeIndex),fo)
        #grid = intr.smooth_grid(11,grid=grid)
        #intr.save_grid(ofile,grid=grid)
        
    #edges = self.graph.edges_from_node_list(self.nodeList)
    #vedges = [e for e in edges if e.index in visited_edges]
    #maxConc = [np.max(e.concentration) for e in vedges]
    #return edges

def main():         
    #dir_ = 'C:\\Users\\simon\\Dropbox\\160113_paul_simulation_results\\LS147T - Post-VDA\\1\\'
    #dir_ = 'C:\\Users\\simon\\Dropbox\\160113_paul_simulation_results\\LS147T\\1\\'
    #f = os.path.join(dir_,'spatialGraph_RIN.am')
    dir_ = 'C:\\Users\\simon\\Dropbox\\Mesentery\\'
    f = dir_ + 'Flow2AmiraPressure.am'

    graph = spatialgraph.SpatialGraph()
    print('Reading graph...')
    graph.read(f)
    print('Graph read')
    
    ia = InjectAgent()
    
    recon = False    
    resume = False
    parallel = True
    largest_inflow = False
    name = 'no reflux'
 
    if recon:
        recon_vascular = True
        recon_interstitium = True
        print 'Reconstructing... Vesels: {} Interstitium {}'.format(recon_vascular,recon_interstitium)
        ia.reconstruct_results(graph,output_directory=dir_,name=name,recon_interstitium=recon_interstitium,recon_vascular=recon_vascular)
    else:
        print 'Simulating...'
        try:
            ia.inject(graph,output_directory=dir_,resume=resume,parallel=parallel,name=name,largest_inflow=largest_inflow)
            print('Simulation complete')
        except KeyboardInterrupt:
            print('Ctrl-C interrupt! Saving graph')
            ia.save_graph(output_directory=dir_)
        
if __name__ == "__main__":
    #import cProfile
    #cProfile.run('main()')
    main()
    #pass
