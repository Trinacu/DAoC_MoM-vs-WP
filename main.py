import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

BASE_CRIT_CHANCE = 0.1
BASE_DMG = 100

COST_TABLE = [1,3,6,10,14,19,25]

def log_tick_formatter(val, pos=None):
   #return f"$10^{{{int(val)}}}$" # remove int() if you don't use MaxNLocator
   #return f"$10^{{{val:.1f}}}$" # remove int() if you don't use MaxNLocator
   return f"${{{10**val:.2f}}}$" # remove int() if you don't use MaxNLocator
   #return f"{10**val:.2e}" #e-Notation

class Player():
   def __init__(self, name='Player', mom_lvl=0, wp_lvl=0):
      self.name = name
      self.mom_lvl = mom_lvl
      self.wp_lvl = wp_lvl
      self.dmg = BASE_DMG * (1 + 0.03 * mom_lvl)
      self.crit_chance = BASE_CRIT_CHANCE + 0.05 * wp_lvl
      
      self.pve_avg_crit_dmg = self.dmg * (0.1 + 0.99)/2
      self.pve_avg_dmg = self.dmg + self.crit_chance * self.pve_avg_crit_dmg
      
      self.rvr_avg_crit_dmg = self.dmg * (0.1 + 0.5)/2
      self.rvr_avg_dmg = self.dmg + self.crit_chance * self.rvr_avg_crit_dmg

   def hit(self, target='pve'):
      a = 'a'

   def print(self):
      print("{}  \tmom:{}\twp:{}\ndmg:{:.1f}\tcrit_chance:{:.2f}".format(self.name, self.mom_lvl, self.wp_lvl, self.dmg, self.crit_chance))
      print("PVE:\tavg_crit_dmg:{:.3f}\tavg_dmg:{:.3f}".format(self.pve_avg_crit_dmg, self.pve_avg_dmg))
      print("RVR:\tavg_crit_dmg:{:.3f}\tavg_dmg:{:.3f}".format(self.rvr_avg_crit_dmg, self.rvr_avg_dmg))

a = Player('Player Mom II WP I', 2, 1)
a.print()

b = Player('Player Mom I WP II', 1, 2)
b.print()

# mastery of magery?, wild power (3% dmg or 5% crit chance)
def rvr_avg_dmg(mom, wp):
   dmg = 100 * (1 + 0.03 * mom)
   return dmg + (0.1 + 0.05 * wp) * dmg * (0.1 + 0.50)/2

def rvr_avg_dmg_increase(mom,wp):
   if mom == 2 and wp == 1:
      print('rvr avg dmg increase: {}'.format(rvr_avg_dmg(mom,wp) - rvr_avg_dmg(0,0)))
   return rvr_avg_dmg(mom,wp) - rvr_avg_dmg(0,0)


def rvr_avg_dmg_increase_per_cost(mom, wp):
   # handle getting array as input so we can plot 3D?
   cost = get_cost(mom, wp)
   
   if cost == 0:
      return 0
   else:
      return rvr_avg_dmg_increase(mom,wp) / cost

def pve_avg_dmg(mom, wp):
   dmg = 100 * (1 + 0.03 * mom)
   return dmg + (0.1 + 0.05 * wp) * dmg * (0.1 + 0.99)/2

def pve_avg_dmg_increase(mom,wp):
   if mom == 2 and wp == 1:
      print('pve avg dmg increase: {}'.format(pve_avg_dmg(mom,wp) - pve_avg_dmg(0,0)))
   return pve_avg_dmg(mom,wp) - pve_avg_dmg(0,0)

def pve_avg_dmg_increase_per_cost(mom,wp):
   cost = get_cost(mom, wp)
   # don't divide by 0
   if cost == 0:
      return 0
   else:
      return pve_avg_dmg_increase(mom,wp) / cost
   

def get_cost(mom, wp):
   rem = (mom % 1)
   mom_lvl = mom - rem
   mom_lvl_int = int(mom_lvl)
   mom_cost = sum(COST_TABLE[:mom_lvl_int]) + rem * COST_TABLE[mom_lvl_int+1]
   
   rem = (wp % 1)
   wp_lvl = wp - rem
   wp_lvl_int = int(wp_lvl)
   wp_cost = sum(COST_TABLE[:wp_lvl_int]) + rem * COST_TABLE[wp_lvl_int+1]
      
      
   cost = mom_cost + wp_cost
   return cost

# 2d PLOT
"""
fig, ax = plt.subplots()
# FIXED MoM LVL
for i in range(6):
   lst = []
   for j in range(6):
      lst.append(rvr_avg_dmg_increase(i, j))
      print(rvr_avg_dmg_increase(i, j))
   ax.plot(lst, ':', label='MoM: '+str(i))
   ax.legend()
# FIXED WP LVL
for i in range(6):
   lst = []
   for j in range(6):
      lst.append(rvr_avg_dmg_increase(j, i))
      print(rvr_avg_dmg_increase(j, i))
   ax.plot(lst, '', label='WP: '+str(i))
   ax.legend()
      
ax.set_xlabel('MoM/WP lvl')
ax.set_ylabel('Average damage increase (RvR) [%]')
ax.set_title('MoM vs WP (RvR)')
plt.show()
"""

# define functions for z axis

def pve_dmg_change(x,y):
   return pve_avg_dmg_increase(x,y)
# vectorize so we can take arrays as inputs (FORCE FLOAT!)
pve_dmg_change = np.vectorize(pve_dmg_change, otypes=[float])

def rvr_dmg_change(x,y):
   return rvr_avg_dmg_increase(x,y)
rvr_dmg_change = np.vectorize(rvr_dmg_change, otypes=[float])

def pve_dmg_change_per_cost(x, y):
   val = pve_avg_dmg_increase_per_cost(x,y)
   return val
pve_dmg_change_per_cost = np.vectorize(pve_dmg_change_per_cost, otypes=[float])

def rvr_dmg_change_per_cost(x, y):
   val = rvr_avg_dmg_increase_per_cost(x,y)
   return val
rvr_dmg_change_per_cost = np.vectorize(rvr_dmg_change_per_cost, otypes=[float])


# define x-y MESH
x = np.linspace(0, 5, 6)
y = np.linspace(0, 5, 6)
X, Y = np.meshgrid(x, y)
Z = pve_dmg_change_per_cost(X, Y)

# WIREFRAME PLOT
"""
# plot wireframe of increased dmg per cost with LOGARITHMIC z-axis
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel('MoM lvl')
ax.set_ylabel('WP lvl')
ax.set_zlabel('Average damage increase per RP cost [%]')
ax.set_title('MoM vs WP (logarithmic z-axis)')

# LOG z-axis!!!
ax.plot_wireframe(X, Y, np.log10(pve_dmg_change_per_cost(X,Y)), rstride=1, cstride=1, color='blue')
ax.plot_wireframe(X, Y, np.log10(rvr_dmg_change_per_cost(X,Y)), rstride=1, cstride=1, color='orange')

# LOG IN tick_formatter!
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# plot points
for i in range(6):
   for j in range(6):
      z = np.log10(pve_dmg_change_per_cost(i,j))
      ax.scatter(i, j, z)
      ax.text(i, j, z,  '{:.2f}'.format(pve_dmg_change_per_cost(i,j)), size=10, zorder=1, color='k')

plt.show()
"""


"""
 SURFACE PLOT (PvE)
"""
"""
# SURFACE PLOT (PvE) LINEAR
# plot points and surface per cost with LINEAR z-axis
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel('MoM lvl')
ax.set_ylabel('WP lvl')
ax.set_zlabel('Average damage increase per RP cost [%/RP]')
ax.set_title('MoM vs WP (PvE) (linear z-axis)')

# POINTS
for i in range(6):
   for j in range(6):
      z = pve_dmg_change_per_cost(i,j)
      ax.scatter(i, j, z)
      ax.text(i, j, z,  '{},{}:{:.2f}'.format(i,j,z), size=10, zorder=1, color='k')

# SURFACE
ax.plot_surface(X, Y, pve_dmg_change_per_cost(X,Y), rstride=1, cstride=1, cmap="plasma", linewidth=0, alpha=0.6, antialiased=True)


plt.show()

# SURFACE PLOT (PvE) LOGARITHMIC
# plot points and surface per cost with LINEAR z-axis
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel('MoM lvl')
ax.set_ylabel('WP lvl')
ax.set_zlabel('Average damage increase per RP cost [%/RP]')
ax.set_title('MoM vs WP (PvE) (logarithmic z-axis)')

# POINTS
for i in range(6):
   for j in range(6):
      z = pve_dmg_change_per_cost(i,j)
      ax.scatter(i, j, 0 if z == 0 else np.log10(z))
      ax.text(i, j, 0 if z == 0 else np.log10(z),  '{},{}:{:.2f}'.format(i,j,z), size=10, zorder=1, color='k')

# SURFACE
Z = pve_dmg_change_per_cost(X,Y)
Z[Z != 0] = np.log10(Z[Z != 0])
Z[0,0] = 0
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="plasma", linewidth=0, alpha=0.6, antialiased=True)
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

# LOG IN tick_formatter!
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.show()
"""

"""
 SURFACE PLOT (RvR)
"""
# SURFACE PLOT (RvR) LINEAR
# plot points and surface per cost with LINEAR z-axis
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel('MoM lvl')
ax.set_ylabel('WP lvl')
ax.set_zlabel('Average damage increase per RP cost [%/RP]')
ax.set_title('MoM vs WP (RvR) (linear z-axis)')

# POINTS
for i in range(6):
   for j in range(6):
      z = rvr_dmg_change_per_cost(i,j)
      ax.scatter(i, j, z)
      ax.text(i, j, z,  '{},{}:{:.2f}'.format(i,j,z), size=10, zorder=1, color='k')

# SURFACE
ax.plot_surface(X, Y, rvr_dmg_change_per_cost(X,Y), rstride=1, cstride=1, cmap="plasma", linewidth=0, alpha=0.6, antialiased=True)


plt.show()


# SURFACE PLOT (RvR) LOGARITHMIC
# plot points and surface per cost with LINEAR z-axis
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.set_xlabel('MoM lvl')
ax.set_ylabel('WP lvl')
ax.set_zlabel('Average damage increase per RP cost [%/RP]')
ax.set_title('MoM vs WP (RvR) (logarithmic z-axis)')

# POINTS
for i in range(6):
   for j in range(6):
      z = rvr_dmg_change_per_cost(i,j)
      ax.scatter(i, j, 0 if z == 0 else np.log10(z))
      ax.text(i, j, 0 if z == 0 else np.log10(z),  '{},{}:{:.2f}'.format(i,j,z), size=10, zorder=1, color='k')

# SURFACE
Z = rvr_dmg_change_per_cost(X,Y)
Z[Z != 0] = np.log10(Z[Z != 0])
Z[0,0] = 0
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="plasma", linewidth=0, alpha=0.6, antialiased=True)
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

# LOG IN tick_formatter!
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.show()


